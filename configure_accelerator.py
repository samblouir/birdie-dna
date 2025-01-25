# configure_accelerator.py
"""
=============================================================================
ACCELERATOR CONFIG SCRIPT
NOTE: If you re-train an existing model, the directory will end in _1, _2, etc.
=============================================================================
"""
import accelerate
import os
import datetime
import utils
from functools import partial

def add_accelerator_and_log_fn(config):
	"""
	Prepares an accelerator for distributed training.
	Adds the accelerator to the config,
	Sets up logging directories,
	"""

	# Sets a logging directory as 'save_dir'
	config_name = config.get("config_name", "CONFIG_NAME_NOT_SET")
	# Save the logs in a directory named after the config
	project_dir = os.path.join(os.path.dirname(__file__), "saves", config_name, f"project")

	## Uncomment to run the same config multiple times with seperate log dirs
	# counter = 0
	# project_dir = os.path.join(os.path.dirname(__file__), "saves", config_name, f"project_{counter}")
	# while os.path.exists(project_dir):
	# 	counter += 1
	# 	project_dir = os.path.join(os.path.dirname(__file__), "saves", config_name, f"project_{counter}")

	os.makedirs(project_dir, exist_ok=True)

	# Optionally store it in config so train.py can access it
	config["project_dir"] = project_dir

	config['log_fn'] = partial(
		utils.log_fn,
		log_dir = project_dir,
	)

	# --------------------------------------------------------
	# Set up the accelerator
	# --------------------------------------------------------
	accelerator_kwargs = dict(
		mixed_precision=config.get("accelerator_mixed_precision_dtype", "bf16"),
		gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
		device_placement=True,
		kwargs_handlers=[
			accelerate.InitProcessGroupKwargs(
				# Can help reduce desyncs at the start of training, even though we have a wait_for_everyone
				# Helpful in case one host is randomly substantially slower to compile than the others
				timeout=datetime.timedelta(seconds=3000) 
			)
		],
		project_dir=config.get("project_dir", None),
		project_config=accelerate.utils.dataclasses.ProjectConfiguration(project_dir=project_dir, automatic_checkpoint_naming=False, logging_dir=project_dir,),
	)
	accelerator = accelerate.Accelerator(**accelerator_kwargs)

	# Update the config with the accelerator.
	config["accelerator"] = accelerator
	config['accelerator_kwargs'] = accelerator_kwargs

	keys_to_add = [
		"device",
		"num_processes",
		"is_main_process",
		"project_dir",
	]
	for key in keys_to_add:
		config[key] = getattr(accelerator, key)

	return config