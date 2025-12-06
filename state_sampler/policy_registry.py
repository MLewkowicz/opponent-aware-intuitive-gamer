"""
Evaluator registry for managing different evaluation metrics.
"""

from typing import Dict, Any

# Registry of available evaluators
SAMPLER_REGISTRY = {
}

def get_sampler(name: str):
    """Get evaluator class by name."""
    if name not in SAMPLER_REGISTRY:
        raise KeyError(f"Unknown sampler: {name}")
    return SAMPLER_REGISTRY[name]

def instantiate_sampler(sampler_config: Dict[str, Any]):
    """Instantiate a sampler from configuration."""
    if isinstance(sampler_config, str):
        sampler_name = sampler_config
        sampler_params = {}
    else:
        sampler_name = sampler_config["name"]
        sampler_params = sampler_config.get("parameters", {})

    SamplerClass = get_sampler(sampler_name)
    return SamplerClass(**sampler_params)