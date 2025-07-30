
class ConfigManager:
    def __init__(self, config_file):
        self.config = self.load_config(config_file
    
    def load_config(self, filename):
        with open(filename, 'r') as f:
            return json.load(f
        