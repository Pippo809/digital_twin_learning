import os

DIGITAL_TWIN_LEARNING_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
"""Absolute path to the legged-gym repository."""

DIGITAL_TWIN_ENVS_DIR = os.path.join(DIGITAL_TWIN_LEARNING_ROOT_DIR, "resources")
"""Absolute path to the module `legged_gym.envs` in legged-gym repository."""

DIGITAL_TWIN_LOG_DIR = os.path.join(DIGITAL_TWIN_LEARNING_ROOT_DIR, "results")
"""Absolute path to the module `legged_gym.envs` in legged-gym repository."""
