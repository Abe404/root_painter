import json
import platform


class Settings:
    def __init__(self):
        self.__config = {}
        for profile in self.get_profiles():
            self.__extend(self.__load_profile(profile))

    def get(self, propname):
        return self.__config.get(propname)

    def get_with_default(self, propname, default_value):
        if self.get(propname) != None:
            return self.get(propname)
        return default_value

    def get_profiles(self):
        return ("base", self.get_current_platform())

    def get_current_platform(self):
        if self.is_mac():
            return "mac"
        if self.is_linux():
            return "linux"
        if self.is_windows():
            return "windows"
        return "unknown"

    def is_mac(self):
        system_name = platform.system()
        return system_name == "Darwin"

    def is_windows(self):
        system_name = platform.system()
        return system_name == "Windows"

    def is_linux(self):
        system_name = platform.system()
        return system_name == "Linux"

    # PRIVATE

    def __extend(self, dict):
        for (key, value) in dict.items():
            self.__config[key] = value

    def __load_profile(self, profile):
        return self.__safe_load_json(f"src/build/settings/{profile}.json")

    def __safe_load_json(self, filename):
        try:
            file = open(filename)
            return json.load(file)
        except FileNotFoundError:
            # If no file, return empty object
            print(f'No config found "{filename}"')
            return {}
