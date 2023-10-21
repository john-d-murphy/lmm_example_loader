import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class Evaluator:
    def __init__(self, arguments):
        self.arguments = arguments

    def run(self):
        self.load_data(self.arguments)
        self.evaluate_model(self.arguments)
        self.show_results(self.arguments)

    def load_data(self, arguments):
        self.loader = DataLoader(
            dataset,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
        )

    def evaluate_model(self, arguments):
        pass

    def show_results(self, arguments):
        pass
