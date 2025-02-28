from common_tools.helpers.duration_helper import DurationHelper
from common_tools.helpers.txt_helper import txt


class ImportHelper:
    @staticmethod
    def test_api_imports_duration(files_paths: list = []):        
        for file_path in files_paths:
            DurationHelper.print_all_imports_duration_for_file(file_path)

class LazyImport:
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name: str) -> any:
        import importlib
        if self.module is None:
            self.module = importlib.import_module(self.module_name)
        return getattr(self.module, name)