class EndWorkflowException(Exception):
    def __init__(self, name: str, message: str):
        self.name = name
        self.message = message