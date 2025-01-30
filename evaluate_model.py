# evaluate_model.py
"""
=============================================================================
EVALUATION SCRIPT
-----------------
1) Loads a config
2) Builds the same model.
3) Finds all checkpoints in saves/<config_name>/checkpoints.
4) Loads the test set from your dataset splits.
5) Evaluates on all of the checkpoints available for the model
=============================================================================
"""

import os
import torch
from tqdm import tqdm
import utils
import configure_accelerator

from functools import partial
from modeling.basemodel import BaseModel
from modeling.tokenizer import DNATokenizer
from data_utils.dataloader import load_dataset, create_clm_batch_generator




def main():
	'''
		Minimal evaluation function
	'''
	try:
		# 1) Parse command-line (like train.py)
		args = utils.parse_args()  # --config=<name>
		config = utils.load_config(args.config)

		# 2) Initialize Accelerator & logging
		config = configure_accelerator.add_accelerator_and_log_fn(config)
		accelerator = config["accelerator"]

		model = BaseModel(config)
		model = torch.compile(model)
		model.to(accelerator.device)
		model = accelerator.prepare(model)

		# Loads in the genome sequence test dataset
		splits = load_dataset(config)
		ds_test = splits["test"]

		
		# Prepare the tokenizer and data pipeline
		tokenizer = DNATokenizer(config)
		config["tokenizer"] = tokenizer
		test_batches = create_clm_batch_generator(
			dataset=ds_test,
			tokenizer=tokenizer,
			max_sequence_length=config["sequence_length"],
			batch_size=config["batch_size"],
			infinite_loop=False,
			split="test",
			number_of_batches=8,
		)

		# Restore the newest checkpoint
		checkpoint_infos = utils.get_checkpoints(config)


		# Clears test_losses.txt only if we are successful
		once = 1
		for ckpt_info in checkpoint_infos:
			ckpt_path = ckpt_info["path"]
			ckpt_step = ckpt_info["step"]

			print(f"\n" * 3, end='',)
			print(f"*" * 60,)

			accelerator.print(f"  Loading {ckpt_step}: {ckpt_path}...")
			accelerator.load_state(ckpt_path)

			# Run evals
			test_loss = utils.evaluate(model=model, batches=test_batches, accelerator=accelerator)
			accelerator.print(f"Finished evaluation. Test Loss = {test_loss:.4f}")

			if once:
				# clear the existing test_losses.txt file
				config["log_fn"]("test_losses.txt", "step,test_loss", write_type="w")
				once = 0

			config["log_fn"]("test_losses.txt", f"{ckpt_step},{test_loss}", write_type="a")
	except Exception as e:
		raise e
	finally:
		accelerator.print("Finished evaluation.")
		accelerator.wait_for_everyone()
		accelerator.print("\n\n   ** Exiting... ** \n\n")
		os._exit(1)


if __name__ == "__main__":
	main()
