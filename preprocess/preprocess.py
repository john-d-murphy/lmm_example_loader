class Preprocessor:
    # Constants
    TRAIN = "train"
    VALIDATE = "val"
    TEST = "test"

    def __init__(self, arguments):
        self.arguments = arguments

    def run(self):
        self.make_output_directories(self.arguments)
        self.read_metadata_file(self.arguments)
        self.encode_and_pickle_data(self.arguments)

    def make_output_directories(self, arguments):
        pass

    def read_metadata_file(self, arguments):
        pass

    def encode_and_pickle_data(self, arguments):
        pass
