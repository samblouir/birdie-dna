# configs/medium.py
"""
=============================================================================
medium CONFIGURATION FILE
=============================================================================
"""

from configs.default import get_config as get_default_config

def get_config():
    """
    Loads in the default config, thne applies these updates

    Returns:
        dict: The modified 'medium' configuration.
    """
    config = get_default_config()
    config.update({
        "config_name": "medium",

        "batch_size": 16,
        "sequence_length": 1024,
        
        # Taken literally for normalization: Each (Attn -> MLP) sequences counts as two layers
        "num_layers": 32,
        "num_steps": 32768,
        "hidden_size": 1024,
        "num_heads": 16,
        "head_dim": 64,
        "eval_interval": 512,
    })
    return config