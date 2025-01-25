# configs/mini.py
"""
=============================================================================
MINI CONFIGURATION FILE
=============================================================================
"""

from configs.default import get_config as get_default_config

def get_config():
    """
    Loads in the default config, thne applies these updates

    Returns:
        dict: The modified 'mini' configuration.
    """
    config = get_default_config()
    config.update({
        "config_name": "mini",
        
        # Taken literally for normalization: Each (Attn -> MLP) sequences counts as two layers
        "num_layers": 4,      
        "num_steps": 1024,
        "eval_interval": 128,
        # "hidden_size": 128,   
    })
    return config