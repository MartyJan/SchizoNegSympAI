import os

import yaml


def read_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config.yaml"
    )
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
