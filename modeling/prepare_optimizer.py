# modeling/prepare_optimizer.py

"""
=============================================================================
PREPARE OPTIMIZER & SCHEDULER
-----------------------------
Defines a function to create AdamW and a cosine-with-warmup scheduler for training.
Excludes embedding parameters from weight decay.
=============================================================================
"""

import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def create_optimizer(model, config):
	"""
	Create an AdamW optimizer plus a simple warmup + cosine decay schedule.

	This version creates separate parameter groups so that:
	  - Embedding parameters have weight_decay=0.0
	  - All other parameters use 'weight_decay' from config (default 0.01).

	Args:
		model (nn.Module): The model to optimize.
		config (dict): Must contain:
			- 'lr': learning rate
			- 'num_steps': total training steps to define warmup & decay
			- 'weight_decay': (optional) weight decay factor, default 0.01

	Returns:
		(optimizer, scheduler)
	"""
	# 1) Build param groups
	no_decay_names = [
		"embeddings",
	]
	weight_decay = config.get("weight_decay", 0.1)

	param_groups = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue  # skip frozen weights

		if any(substr in name.lower() for substr in no_decay_names):
			# Group with weight_decay=0.0
			param_groups.append({
				"params": [param],
				"weight_decay": 0.0,
			})
			config['log_fn']("param_groups.txt", f"param_group: {name}, weight_decay: {weight_decay}")
		else:
			# Group with normal weight decay
			param_groups.append({
				"params": [param],
				"weight_decay": weight_decay,
			})
			config['log_fn']("param_groups.txt", f"param_group: {name}, weight_decay: {weight_decay}")

	# 2) Create optimizer from param_groups
	optimizer = AdamW(param_groups, lr=config.get("lr", 1e-3))

	# 3) Create a warmup + cosine decay schedule
	num_steps = config["num_steps"]
	warmup_steps = int(0.1 * num_steps)  # 10% warmup by default

	def lr_lambda(step):
		if step < warmup_steps:
			# Linear warmup from 0 to 1
			return float(step) / float(max(1, warmup_steps))
		else:
			# Cosine decay from 1 down to 0
			progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
			return 0.5 * (1.0 + math.cos(math.pi * progress))

	scheduler = LambdaLR(optimizer, lr_lambda)

	return optimizer, scheduler
