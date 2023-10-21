#! /usr/bin/python

#### Imports
# External Libraries
import argparse
import hashlib
import logging
import os
import sys
import textwrap

from preprocess.preprocess import Preprocessor
from train.train import Trainer
from evaluate.evaluate import Evaluator
from generate.generate import Generator

#### Constants


#### Logger
log = logging.getLogger("root")
LOG_FORMAT = "[%(asctime)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
log.setLevel(logging.DEBUG)


def main():
    ### Parse and display arguments
    arguments = parse_arguments()

    ### Preprocess Data
    if arguments.preprocess:
        log.info("Running Preprocessor")
        preprocessor = Preprocessor(arguments)
        preprocessor.run()

    ### Train Data
    if arguments.train:
        log.info("Running Trainer")
        trainer = Trainer(arguments)
        trainer.run()

    ### Evaluate Data
    if arguments.evaluate:
        log.info("Running Evaluate")
        evaluator = Evaluator(arguments)
        evaluator.run()

    ### Generate New Data
    if arguments.generate:
        log.info("Running Generate")
        generator = Generator(arguments)
        generator.run()


def parse_arguments():
    ### Get Arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=textwrap.dedent("Description of the program."),
        epilog="Summary/notes/etc.",
    )

    preprocess = parser.add_argument_group(
        "Preprocessing Options",
        "Details on the Preprocessor, what to expect, etc.",
    )

    preprocess.add_argument(
        "-p",
        "--preprocess",
        default=False,
        help="Preprocess Data",
        action="store_true",
    )

    preprocess.add_argument(
        "-pi",
        "--preprocess_input_directory",
        metavar="\b",
        default=None,
        help="Preprocessor Input Directory",
    )

    preprocess.add_argument(
        "-po",
        "--preprocess_output_dir",
        metavar="\b",
        default=None,
        help="Preprocessor Output Directory",
    )

    train = parser.add_argument_group(
        "Training Options",
        "Details on the Training Options, what to expect, etc.",
    )

    train.add_argument(
        "-t",
        "--train",
        default=False,
        help="Train On Data",
        action="store_true",
    )

    train.add_argument(
        "--use_tensorboard",
        default=False,
        help="Use a Tensorboard to Show Progress of Training",
        action="store_true",
    )

    evaluate = parser.add_argument_group(
        "Evaluation Options",
        "Details on the Evaluation Options, what to expect, etc.",
    )
    evaluate.add_argument(
        "-e",
        "--evaluate",
        default=False,
        help="Evaluate Data",
        action="store_true",
    )

    evaluate.add_argument(
        "--batch_size",
        default=128,
        metavar="\b",
        help="Batch Size for Evaluation Neural Network",
        type=int,
    )

    evaluate.add_argument(
        "--num_workers",
        default=16,
        metavar="\b",
        help="Number of Workers for Evaluation Neural Network",
        type=int,
    )

    generate = parser.add_argument_group(
        "Generation Options",
        "Details on the Generation Options, what to expect, etc.",
    )
    generate.add_argument(
        "-g",
        "--generate",
        default=False,
        help="Generate New Output",
        action="store_true",
    )

    arguments = parser.parse_args()
    log.info("Preprocess            - %s" % arguments.preprocess)
    log.info("Preprocess Input Dir  - %s" % arguments.preprocess_input_directory)
    log.info("Preprocess Output Dir - %s" % arguments.preprocess_output_dir)
    log.info("Train                 - %s" % arguments.train)
    log.info("Evaluate              - %s" % arguments.evaluate)
    log.info("Generate              - %s" % arguments.generate)

    return arguments


if __name__ == "__main__":
    main()
