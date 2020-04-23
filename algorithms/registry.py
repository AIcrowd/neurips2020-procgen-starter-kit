"""
Registry of custom implemented algorithms names
"""

def _import_custom_random_agent():
    from .custom_random_agent.custom_random_agent import CustomRandomAgent


CUSTOM_ALGORITHMS = {
    "custom/CustomRandomAgent" : _import_custom_random_agent
}