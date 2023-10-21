class Trainer:
    def __init__(self, arguments):
        self.arguments = arguments

    def run(self):
        self.create_output_directories(self.arguments)
        self.create_tensorboard(self.arguments)
        self.load_datasets(self.arguments)
        self.train_model(self.arguments)
        self.report_results(self.arguments)

    def create_output_directories(self, arguments):
        pass

    def create_tensorboard(self, arguments):
        pass

    def load_datasets(self, arguments):
        pass

    def train_model(self, arguments):
        pass

    def report_results(self, arguments):
        pass
