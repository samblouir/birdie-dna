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

        "batch_size": 16,
        "sequence_length": 1024,
        
        # Taken literally for normalization: Each (Attn -> MLP) sequences counts as two layers
        "num_layers": 8,
        "num_steps": 32768,
        "hidden_size": 768,
        "num_heads": 12,
        "head_dim": 64,
        "eval_interval": 512,
    })
    return config