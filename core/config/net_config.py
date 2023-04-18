import json


class _NetConfig:
    def __init__(self) -> None:
        with open('net_config.json') as config_file:
            self.config_file = json.load(config_file)

    @property
    def key_point_classifier_model_path(self) -> str:
        return self.config_file['key_point_classifier_model_path']

    @property
    def names_path(self) -> str:
        return self.config_file['names_path']


NetConfig = _NetConfig()
