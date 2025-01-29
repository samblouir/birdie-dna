# configs/bert.py
"""
=============================================================================
BERT NORMAL CONFIGURATION FILE
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
        "config_name": "micro",

        "batch_size": 16,
        "sequence_length": 1024,
        
        # Taken literally for normalization: Each (Attn -> MLP) sequences counts as two layers
        "num_layers": 4,
        "num_steps": 32768,
        "hidden_size": 512,
        "num_heads": 8,
        "head_dim": 64,
        "eval_interval": 512,
    })
    return config