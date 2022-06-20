import json


class _NetConfig:
    def __init__(self) -> None:
        with open('net_config.json') as config_file:
            self.config_file = json.load(config_file)

    @property
    def weights_path(self) -> str:
        return self.config_file['weights_path']

    @property
    def cfg_path(self) -> str:
        return self.config_file['cfg_path']

    @property
    def names_path(self) -> str:
        return self.config_file['names_path']


NetConfig = _NetConfig()
