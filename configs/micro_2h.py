# configs/micro_2h.py
import configs.micro as micro_cfg
"""
=============================================================================
micro_2h CONFIGURATION FILE
=============================================================================
"""

from configs.default import get_config as get_default_config

def get_config():
	"""
	Loads in the default config, thne applies these updates

	Returns:
		dict: The modified 'micro_2h' configuration.
	"""
	config = get_default_config()
	
	config.update({
		**micro_cfg.get_config(),
		"config_name": "micro_2h",
		"num_heads": 2,
	})
	return config