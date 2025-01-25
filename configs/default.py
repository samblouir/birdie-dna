# configs/default.py
"""
=============================================================================
DEFAULT CONFIGURATION FILE
--------------------------
Provides a dictionary of baseline hyperparameters used throughout training.
Each key is explained line-by-line below.
=============================================================================
"""
import utils

def calculate_num_heads(config):
	num_heads = config.get("num_heads", None)
	if num_heads is None:
		num_heads = config['hidden_size'] // config['head_dim']
	return num_heads


def get_config():
	"""
	Returns a dictionary of default training hyperparameters.
	"""
	# Create the config dictionary with default values
	
	# TODO: Organize these.
	config = {
		"config_name": "default",


		"sequence_length": 1024,
		"validation_batch_count": 32,
		"batch_size": 32,
		"num_steps": 32_768,
		"eval_interval": 512,
		"save_interval": 512,

		"hidden_size": 512,
		"head_dim": 64,
		"num_heads": 8, # 
		"num_gqa_heads": None,  # CURRENTLY DISABLED - seen in functions below

		# Taken literally for normalization: Each (Attn -> MLP) sequences counts as two layers
		"num_layers": 16,

		"gradient_accumulation_steps": 1, # Not implemented yet 

		"dataset_np_rng_seed": 0, # Used for shuffling the datasets and determining what data is used for train/validation/test splits
		"projection_layers_use_bias": False,
		"num_workers_dataset": 16,
		"clip_grad_norm": 1.0,
		"accelerator_mixed_precision_dtype": "bf16",
		"lr": 3e-4,
		# for parameters
		"seed": 42,
		"vocab_size": 512,

		# Currently not applied to embeddings
		"weight_decay": 0.1,
		"dtype": "bf16",
		"param_dtype": "fp32",

		 # Currently disabled               
		"use_fusedlce": 0,

		"base_decay_rate": 500_000,
		"pretraining_sequence_length": 1024,
		"use_attention": True,
		"use_mlp": True,
		"mlp_dim_mult": 4.0,

		
		'''
			The functions will calculate their values
		'''
		"functions": {

			"dtype_str": lambda x: utils.str_to_dtype(x['dtype']),
			"dtype": lambda x: utils.str_to_dtype(x['dtype']),
			"param_dtype": lambda x: utils.str_to_dtype(x['dtype']),

			"num_heads": lambda x: calculate_num_heads(x),
			"num_gqa_heads": lambda x: calculate_num_heads(x),
		},
		
	}
	
	for function in config.get('functions', []):
		config[function] = config['functions'][function](config)

	return config