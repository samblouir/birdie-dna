# config.py
"""
=============================================================================
GLOBAL CONFIG FLAGS
-------------------
enable_global_debug_print: setting this to != 0 enables debug prints in utils.py
=============================================================================
"""
import os
import time
rank = os.environ.get("RANK", None)
accelerate_index = os.environ.get("ACCELERATE_PROCESS_INDEX", None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "9"
os.environ['WANDB_LOG_MODEL'] = "false"
os.environ['TORCHDYNAMO_VERBOSE'] = "1"
####################################################################
####################################################################
####################################################################
####################################################################
## Global debug prints
enable_global_debug_print = 0
enable_global_debug_print = 1
####################################################################
is_main_process = (rank or accelerate_index) == "0"
if not is_main_process:
	enable_global_debug_print = 1
####################################################################
####################################################################
####################################################################


# Stores the current time for logging purposes.
start_time = time.time()





## Prints out a message
if enable_global_debug_print:
	print(f"  [DEBUG] {__file__}: Global debug prints enabled. To disable it, set enable_global_debug_print to 0 or False in this file.")
