# train.py
"""
=============================================================================
TRAINING SCRIPT USING ACCELERATE
-------------------------------
 - Parses a config name (e.g., --config=mini)
 - Loads the dataset (auto-downloads, if needed)
 - Builds BaseModel (from basemodel.py)
 - Trains for the specified number of steps with AdamW + linear warmup
 - Periodically evaluates on the validation split
 - Afterwards, evaluates on the test split
=============================================================================
"""

import torch
from tqdm import tqdm
from accelerate import Accelerator
import configure_accelerator
from functools import partial
import os
import utils
import config
import pickle
import dill


from modeling.basemodel import BaseModel
from modeling.prepare_optimizer import create_optimizer
from modeling.tokenizer import DNATokenizer

# Consolidated data loading
from data_utils.dataloader import load_dataset, create_clm_batch_generator






def prepare_data_pipelines(config, ds_train, ds_val, *args, **kwargs):
	'''
		Just adds tokenization and batching to the data pipeline
	'''
	data_processer_pipeline = partial(
		create_clm_batch_generator,
		tokenizer=config['tokenizer'],
		max_sequence_length=config["sequence_length"],
		batch_size=config["batch_size"],
		process_idx=config['accelerator'].process_index,
		num_processes=config['accelerator'].num_processes,
		num_workers=config.get("num_workers_dataset", 8),
	)

	ds_val_batched = data_processer_pipeline(
		dataset = ds_val,
		split = "validation",
		number_of_batches = config.get("validation_batch_count", 16),
		infinite_loop = False,
	)

	ds_train_gen = data_processer_pipeline(
		dataset = ds_train,
		split = "train",
		infinite_loop = True,
	)

	return dict(
		ds_train_gen = ds_train_gen,
		ds_val_batched = ds_val_batched,
	)


def train_step(
		model,
		batch,
		optimizer=None,
		scheduler=None,
		accelerator=None,
		config=None,
		return_loss=False,
		*args,
		**kwargs,
	):
	'''
		Train step for the model
	'''
	with accelerator.autocast():
		optimizer.zero_grad()
		loss = model(**batch)
		accelerator.backward(loss)
		accelerator.clip_grad_norm_(model.parameters(), max_norm=config["clip_grad_norm"])
		optimizer.step()
		scheduler.step()

	if return_loss:
		return loss.detach().cpu()
	
	return None




def main():
	args = utils.parse_args()
	config = utils.load_config(args.config)
	config = configure_accelerator.add_accelerator_and_log_fn(config)
	accelerator = config["accelerator"]

	# For param init values
	torch.manual_seed(config["seed"])

	# Build model
	model = BaseModel(config)
	model = torch.compile(model)
	model = model.to(accelerator.device)
	# Save the param count
	param_count = utils.get_model_param_count(model)
	config['log_fn']("model_params.txt", f"param_count: {param_count}")

	# Create optimizer & scheduler
	optimizer, scheduler = create_optimizer(model, config)

	# Load dataset (train, validation, test)
	splits = load_dataset(config)
	ds_train = splits["train"]
	ds_val   = splits["validation"]
	ds_test  = splits["test"]

	# Create tokenizer & data generators
	tokenizer = DNATokenizer(config)
	config['tokenizer'] = tokenizer

	ds_pipelines = prepare_data_pipelines(config, ds_train, ds_val)
	ds_train_gen = ds_pipelines['ds_train_gen']
	ds_val_batched = ds_pipelines['ds_val_batched']

	validation_data_hash = utils.obj_to_str_hash(ds_val_batched)
	config["log_fn"]("validation_data_hash.txt", f"Validation data hash: {validation_data_hash}")
	
	# Backup the config
	with open(os.path.join(config['project_dir'], "config.pkl"), "wb") as f:
		dill.dump(config, f)


	# Wait for everyone to be ready
	accelerator.wait_for_everyone()

	# Prepare model & optimizer with Accelerator
	model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

	# Training loop
	num_steps = config["num_steps"]
	log_loss_interval = config.get("log_loss_interval", 1)
	eval_interval = config["eval_interval"]
	save_interval = config.get("save_interval", eval_interval)
	checkpoint_dir = config.get("checkpoint_dir", "checkpoints")



	model.train()

	# Convenience function for evaluation
	eval_fn = partial(
		utils.evaluate,
		batches=ds_val_batched,
		config=config,
	)

	# Initialize validation loss
	val_loss = eval_fn(model, 0)

	# Wait for everyone to be ready
	accelerator.wait_for_everyone()

	# Attempts to make a save, if we haven't made one yet
	# Create a checkpoint path, e.g.: "./saves/mini/checkpoints/checkpoint_8192"
	_step_idx = 0
	current_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_{int(_step_idx)}")
	if not os.path.exists(current_checkpoint_dir):
		os.makedirs(current_checkpoint_dir, exist_ok=True)
		utils.debug_print(f"  Saving checkpoint at step_idx: {_step_idx:,} to:\n\t\"{current_checkpoint_dir}\"")
		accelerator.save_state(current_checkpoint_dir)

	# Loads the model from the last checkpoint, if we have one.
	last_checkpoint_info = utils.get_checkpoints(config)[-1]
	ckpt_path = last_checkpoint_info["path"]
	ckpt_step = last_checkpoint_info["step"]
	if 0 < ckpt_step:
		accelerator.print(f"  Loading {ckpt_step}: {ckpt_path}...")
		accelerator.load_state(checkpoint_dir)
		accelerator.print(f"  Fast-forwarding the dataset to step_idx: {ckpt_step}...")
		for _ in tqdm(range(ckpt_step)):
			next(ds_train_gen)
		step_idx = ckpt_step
	else:
		step_idx = 0

	## Validate train_batch shape
	dummy_batch = None

	progress_bar = tqdm(range(step_idx, num_steps + 1), disable=not accelerator.is_local_main_process)
	
	for step_idx in progress_bar:
		# Get next batch

		# utils.debug_print(f"  step_idx: {step_idx}")
		batch = next(ds_train_gen)
		# utils.debug_print(f" Acquired batch")
		batch = utils.move_batch_to_device(batch, device=accelerator.device)
		# utils.debug_print(f" Moved batch to device")
		

		maybe_loss = train_step(
			model=model,
			batch=batch,
			optimizer=optimizer,
			scheduler=scheduler,
			accelerator=accelerator,
			config=config,
			return_loss= (step_idx % log_loss_interval)==0,
		)
		# utils.debug_print(f" Completed a training step")

		if (step_idx % log_loss_interval) == 0:
			loss = maybe_loss
			batch_input_ids_shape = batch['input_ids'].shape
			status_str = f"  step_idx: {step_idx}, loss: {loss.item():.4f}, val_loss: {val_loss:0.4f}, batch_input_ids_shape: {batch_input_ids_shape}, param_count: {param_count:,}, lr: {scheduler.get_last_lr()[0]:.4e}"
			progress_bar.set_description(status_str)
			config['log_fn']("training_losses.txt", status_str)


		if ((step_idx % eval_interval) == 0):
			val_loss = eval_fn(model, step_idx)
			status_str = f"  step_idx: {step_idx}, loss: {loss.item():.4f}, val_loss: {val_loss:0.4f}, batch_input_ids_shape: {batch_input_ids_shape}, param_count: {param_count:,}, lr: {scheduler.get_last_lr()[0]:.4e}"
			progress_bar.set_description(status_str)
			config['log_fn']("validation_losses.txt", status_str)

			model_stats = utils.get_model_param_stats(model)
			model_stats_str = f"step_idx: {step_idx}\n{model_stats}\n\n"
			config['log_fn']("model_stats.txt", model_stats_str)
			accelerator.print(model_stats_str)
			model.train()


		if step_idx % save_interval == 0:
			# Create a checkpoint path, e.g.: "./saves/mini/checkpoints/checkpoint_8192"
			current_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_{int(step_idx)}")
			os.makedirs(current_checkpoint_dir, exist_ok=True)
			accelerator.print(f"  Saving checkpoint at step_idx: {step_idx:,} to:\n\t\"{current_checkpoint_dir}\"")
			accelerator.save_state(current_checkpoint_dir)
			utils.debug_print(f"  Saved at step_idx: {step_idx:,}")
			print(f"\n")

	accelerator.print("Training complete.")

	# Closes dataloading processes and threads
	os._exit(1)



if __name__ == "__main__":
	main()