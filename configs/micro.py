# configs/micro.py
import configs.mini as mini_cfg
"""
=============================================================================
MICRO CONFIGURATION FILE
=============================================================================
"""

from configs.default import get_config as get_default_config

def get_config():
	"""
	Loads in the default config, thne applies these updates

	Returns:
		dict: The modified 'micro' configuration.
	"""
	config = get_default_config()
	
	config.update({
		**mini_cfg.get_config(),
		"config_name": "micro",
		"num_layers": 4,
		"hidden_size": 512,
		"num_heads": 8,
		"head_dim": 64,
	})
	return config