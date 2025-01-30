# utils.py
"""
=============================================================================
UTILITY FUNCTIONS
-----------------

This module contains convenient utility functions called all throughout the codebase.
Features include:
  - Parsing command-line arguments
  - Converting dtype strings to real Torch dtypes
  - For ints:
    - Rounding up or down to the nearest divisor.
	- Rounding up or down to the nearest power of two.
  - Converting objects to 32-bit ints (intended for RNG seeding)
  - Saving and loading Python objects using pickle.
  - Logging strings to a specified directory and filename.
  - Sorting dictionaries by key.

Functions:

  - parse_args()
      Parse command-line arguments, specifically --config <name>. Defaults to 'default'.

  - str_to_dtype(dtype_str)
      Convert a string dtype to the corresponding PyTorch dtype.

  - make_divisible_by(x, divisor, round_up=True)
      Ensure 'x' is divisible by 'divisor' by rounding up or down.

  - make_power_of_2(x, round_up=True)
      Adjust 'x' to the nearest power of two.

  - obj_to_int_32(obj)
      Convert any object to a 32-bit hash for seeds.

  - debug_print(...)
      Print debug info if config.enable_global_debug_print is True.

  - quick_save(obj, name)
      Pickle and save 'obj' to a cache file named '{name}.cache' in TMP_DIR.

  - quick_load(name)
      Load and return the object from '{name}.cache' in TMP_DIR.

  - log_fn(filename, result, log_dir=None, write_type='a')
      Helper to log content to a file. The log directory must be specified.

  - sort_dict(d, in_place=False)
      Sort a dictionary by key.
=============================================================================
"""


import argparse
import hashlib
import inspect
import os
import config
import torch
import pickle
import threading
import time
import numpy as np
from tqdm import tqdm



# Returns the checkpoints for a config
def get_checkpoints(config) -> list:
	"""
	Searches in:
	   saves/[project_dir]/[checkpoint_dir]/checkpoint_*
	Returns the ascendingly sorted list of subdirectories.
	"""

	checkpoint_dir = config["checkpoint_dir"]

	if not os.path.exists(checkpoint_dir):
		raise FileNotFoundError(f"No checkpoints directory found at '{checkpoint_dir}'")

	# Look for subfolders named 'checkpoint_<step>'
	subdirs = []
	for name in os.listdir(checkpoint_dir):
		checkpoint_dir = os.path.join(checkpoint_dir, name)
		if os.path.isdir(checkpoint_dir) and name.startswith("checkpoint_"):
			subdirs.append(checkpoint_dir)

	if not subdirs:
		raise FileNotFoundError(f"No subfolders named 'checkpoint_*' found in '{checkpoint_dir}'. Please make sure you have checkpoints here.")

	# Parse out the step number from 'checkpoint_XXXX' to find the newest
	def get_step_num(path: str) -> int:
		basename = os.path.basename(path)
		# e.g. "checkpoint_8192" -> 8192
		return int(basename.split("_")[-1])
	
	# Sort by ascending by step number
	subdirs.sort(key=lambda p: get_step_num(p))
	subdirs = [
		dict(
			path=subdir,
			step=get_step_num(subdir),
		)
		for subdir in subdirs
	]
	return subdirs



# Load a config file from 'configs/[name].py'
def load_config(name: str) -> dict:
	"""
		Imports the config at 'configs/[name].py'
		Calls the 'get_config' function in that py.
	"""
	
	module = __import__(f"configs.{name}", fromlist=["get_config"])
	get_config = getattr(module, "get_config")

	return get_config()

# Move a batch to the specified device.
def move_batch_to_device(batch, device=None, requires_grad=False,):
	"""
	Moves each item in 'batch' to the specified device.
	By default, this has (input_ids, label_ids, segment_ids) as np.arrays being converted to torch Tensors.
	"""
	for key in batch:
		if isinstance(batch[key], torch.Tensor):
			batch[key] = batch[key].to(device).requires_grad_(requires_grad)
		else:
			batch[key] = torch.tensor(batch[key], dtype=torch.long, device=device, requires_grad=requires_grad)
	return batch


# Evaluates the model on the provided batches.
def evaluate(model, step_idx=0, batches=None, config=None, accelerator=None, *args, **kwargs,):
	"""
	Requires ((batches) AND (config OR accelerator)).
	Evaluates on the provided batches.

	Minor quirk: Averaging here does not account for different amounts of padding in different batches - but that is fine for now.
	"""
	assert(batches is not None)

	if (accelerator is None):
		accelerator = config['accelerator']

	model.eval()
	total_loss = 0.0

	progress_bar = tqdm(len(batches), disable=not accelerator.is_local_main_process, desc=f"Evaluating (step_idx: {step_idx:,})")
	with torch.no_grad():
		for batch in batches:
			batch = move_batch_to_device(batch, device=accelerator.device)
			loss = model(**batch)
			total_loss += loss.mean().item()
			progress_bar.update(1)

	gathered_tensors = accelerator.gather(torch.tensor([total_loss], device=accelerator.device))
	global_loss_sum = gathered_tensors.sum().item()
	global_batch_count = (len(batches) * accelerator.num_processes)

	validation_loss = (global_loss_sum / global_batch_count)

	return validation_loss



# Attempt to determine a writable temporary directory.
# If /tmp is writable, use that. Otherwise, create a local 'tmp' directory.
try:
	with open("/tmp/test.txt", "w") as f:
		f.write("test")
	os.remove("/tmp/test.txt")
	TMP_DIR = "/tmp"
except OSError:
	script_dir = os.path.abspath(os.path.dirname(__file__))
	TMP_DIR = os.path.join(script_dir, "tmp")
	os.makedirs(TMP_DIR, exist_ok=True)
	print(f"WARNING: Set TMP_DIR to {TMP_DIR}, since /tmp was not writable.", flush=True)


def parse_args():
	"""
	Parse command-line arguments, specifically the --config <name> parameter.
	Defaults to 'default' if not provided.

	Returns:
		argparse.Namespace: The parsed arguments, including the .config attribute.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--config",
		type=str,
		default="default",
		help="Name of the config file to load (found at config/[name].py)."
	)

	parser.add_argument(
		"--finetune",
		type=str,
		default="False",
		help="Whether to finetune the model or not."
	)
	return parser.parse_args()


def quick_save(key=None, obj=None):
	"""
	Serialize (pickle) and save 'obj' into TMP_DIR under the filename '{key}.cache'.

	Args:
		obj (Any): The Python object to be pickled.
		key (str): The base name of the cache file.

	Returns:
		None
	"""
	path = os.path.join(TMP_DIR, f"{key}.cache")
	os.makedirs(TMP_DIR, exist_ok=True)
	with open(path, 'wb') as f:
		pickle.dump(obj, f)
	debug_print(f"Saved to {path}!")


def quick_load(name):
	"""
	Load and return the object stored in TMP_DIR under '{name}.cache'.

	Args:
		name (str): The base name (without extension) of the cache file.

	Returns:
		Any: The Python object that was previously saved.

	Raises:
		FileNotFoundError: If the cache file does not exist.
	"""
	path = os.path.join(TMP_DIR, f"{name}.cache")
	if not os.path.exists(path):
		raise FileNotFoundError(f"{path} does not exist!")
	with open(path, 'rb') as f:
		return pickle.load(f)




def log_fn(filename, result, log_dir=None, write_type='a', blocking=False,):
	"""
	Log 'result' to a file named 'filename' in the directory 'log_dir'.

	Args:
		filename (str): Name of the log file to write to.
		result (str): The content to append or write.
		log_dir (str): The directory path where the log file will be created.
		write_type (str): File write mode ('a' for append, 'w' for overwrite, etc.).
		blocking (bool): If True, waits for the write operation to complete before returning.


	Returns:
		str: The full path of the created or appended log file.

	Raises:
		ValueError: If log_dir is not provided.
	"""
	if log_dir is None:
		raise ValueError("FATAL EXCEPTION: log_dir must be specified when calling log_fn.")

	os.makedirs(log_dir, exist_ok=True)
	log_path = os.path.join(log_dir, filename)

	if result is not None:
		def run_in_thread():
			with open(log_path, write_type) as f:
				f.write(result + "\n")
				# debug_print(f"Wrote result to {log_path} with write_type={write_type}")
		if blocking:
			run_in_thread()
		else:
			threading.Thread(target=run_in_thread).start()

	return log_path

def get_model_param_count(model):
	"""
	Returns the number of parameters in a model.
	"""
	return sum(p.numel() for p in model.parameters())

def get_model_param_stats(model):
	"""
	Returns model weight stats in String form
	"""
	stat_str = []
	for name, param in model.named_parameters():
		stats = {
			"name": f"{name:>50}",
			"shape": tuple(param.shape),
			"minval": f"{param.min().item():.4f}",
			"maxval": f"{param.max().item():.4f}",
			"mean": f"{param.mean().item():.4f}",
			"std": f"{param.std().item():.4e}",
		}
		stats_strs = [f"{k}: {str(v):>12}" for k, v in stats.items()]
		stat_str.append(', '.join(stats_strs))
	return "\n".join(stat_str)


def str_to_dtype(dtype_str: str):
	"""
	Convert a string dtype to the corresponding PyTorch dtype.

	Supported string representations:
		'bf16', 'bfloat16' -> torch.bfloat16
		'float32', 'fp32' -> torch.float32
		'float64', 'fp64' -> torch.float64
		'int32'          -> torch.int32
		'int64'          -> torch.int64

	Args:
		dtype_str (str): String representing a PyTorch dtype. If not a string, returns the input unchanged.

	Returns:
		torch.dtype: The corresponding PyTorch dtype.

	Raises:
		ValueError: If the string is not recognized as a valid dtype.
	"""
	if not isinstance(dtype_str, str):
		return dtype_str
	dtype_str = dtype_str.lower()
	if dtype_str in ['bf16', 'bfloat16']:
		return torch.bfloat16
	elif dtype_str in ['float32', 'fp32']:
		return torch.float32
	elif dtype_str in ['float64', 'fp64']:
		return torch.float64
	elif dtype_str in ['int32']:
		return torch.int32
	elif dtype_str in ['int64']:
		return torch.int64
	else:
		debug_print(f"Invalid dtype string: {dtype_str}")
		raise ValueError(f"  dtype string: {dtype_str}")


def make_divisible_by(x: int, divisor: int, round_up: bool = True) -> int:
	"""
	Ensures 'x' is divisible by 'divisor' by either rounding up or down.

	Example:
		make_divisible_by(10, 8, round_up=False) -> 8
		make_divisible_by(10, 8, round_up=True)  -> 16

	Args:
		x (int): The integer to adjust.
		divisor (int): The divisor.
		round_up (bool): If True, rounds up to the nearest multiple; otherwise rounds down.

	Returns:
		int: The adjusted value of 'x' that is divisible by 'divisor'.

	Raises:
		ValueError: If 'divisor' is zero.
	"""
	if divisor == 0:
		raise ValueError("Divisor cannot be zero.")
	remainder = x % divisor
	if remainder == 0:
		return x
	return x + (divisor - remainder) if round_up else x - remainder


def make_power_of_2(x: int, round_up: bool = True) -> int:
	"""
	Adjust 'x' to the nearest power of two (either up or down).

	Example:
		make_power_of_2(10, round_up=True)  -> 16
		make_power_of_2(10, round_up=False) -> 8

	Args:
		x (int): The integer to adjust.
		round_up (bool): If True, rounds up to the nearest power of 2; otherwise down.

	Returns:
		int: The power-of-two-adjusted integer.
	"""
	x = int(x)
	if x < 1:
		return 1
	if round_up:
		return 2 ** (x - 1).bit_length()
	else:
		return 2 ** ((x).bit_length() - 1)


def obj_to_int_32(obj):
	"""
	Convert any object to a 32-bit integer hash via SHA-256 of its string form.
	Useful for reproducible seeds from arbitrary Python objects.

	Args:
		obj (Any): The object to be hashed.

	Returns:
		int: The lower 32 bits of the SHA-256 hash of the object's string representation.
	"""
	obj_str = str(obj)
	hash_str = hashlib.sha256(obj_str.encode('utf-8')).hexdigest()
	# Take lower 32 bits
	seed_int = int(hash_str, 16) & 0xFFFFFFFF
	return seed_int


def obj_to_str_hash(obj):
	"""
	Convert any object to a full SHA-256 hash string of its string form.

	Args:
		obj (Any): The object to be hashed.

	Returns:
		str: Hex digest string representing SHA-256 hash.
	"""
	obj_str = str(obj)
	hash_str = hashlib.sha256(obj_str.encode('utf-8')).hexdigest()
	return hash_str


def debug_print(*args, **kwargs):
	"""
	Conditionally prints debugging messages if config.enable_global_debug_print is True.
	Also prints the caller's file/function for clarity.

	Usage like print(), but can pass 'is_main_process' etc. as a keyword arg.

	Args:
		*args: Positional arguments to be printed.
		**kwargs: Keyword arguments for print(). 'is_main_process' (bool) can be passed to control printing.
	"""
	caller_frame = inspect.stack()[1]
	caller_file_name = os.path.basename(caller_frame.filename)
	fn_name = caller_frame.function

	is_main_process = kwargs.pop("is_main_process", True)
	elapsed_time = (time.time() - config.start_time)
	if config.enable_global_debug_print and is_main_process:
		print(f"  [DEBUG] ({np.round(elapsed_time,4):,}): {caller_file_name}:{fn_name}():", *args, **kwargs)


def sort_dict(d: dict, in_place: bool = False) -> dict:
	"""
	Sort a dictionary by its keys and return a new dictionary, or optionally sort in-place.

	Args:
		d (dict): The dictionary to sort.
		in_place (bool): If True, sort the dictionary in-place and return the same object.

	Returns:
		dict: The sorted dictionary (or the same dictionary if in_place=True).
	"""
	if in_place:
		temp_dict = sort_dict(d, in_place=False)
		d.clear()
		d.update(temp_dict)
		return d

	return {k: d[k] for k in sorted(d)}


