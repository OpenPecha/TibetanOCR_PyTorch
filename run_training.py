"""
A simple interface for OCR training.

use e.g. python run_training.py --input_dir "Datasets/KhyentseWangpo"
"""

import os
import sys
import logging
import argparse
from glob import glob
from natsort import natsorted
from config import N_DEFAULT_CHARSET
from src.Modules import CRNNNetwork, EasterNetwork
from src.Modules import OCRTrainer
from src.Utils import create_dir


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--architecture",
        Choices=["CRNN", "Easter2"],
        required=False,
        default="Easter2",
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument(
        "-o", "--output_dir", type=str, required=False, default="Output"
    )
    parser.add_argument("-l", "--file_limit", type=int, required=False, default=-1)
    parser.add_argument(
        "-f",
        "--label_format",
        choices=["Tibetan", "Wylie"],
        required=False,
        default="Tibetan",
    )
    parser.add_argument("-b", "--batch_size", type=int, required=False, default=32)
    parser.add_argument("-t", "--time_steps", type=int, required=False, default=500)
    parser.add_argument("-e", "--epochs", type=int, required=False, default=30)
    parser.add_argument(
        "-o", "--optimizer", choices=["Adam", "RMSProp"], required=False, default="Adam"
    )
    parser.add_argument(
        "-n", "--model_name", type=str, required=False, default="ocr_model"
    )

    args = parser.parse_args()

    architecture = args.architecture
    input_dir = args.input_dir
    output_dir = args.output
    file_limit = args.file_limit
    label_format = args.label_format
    batch_size = args.batch_size
    time_steps = args.time_steps
    epochs = args.epochs
    optimizer = args.optimizer
    model_name = args.model_name

    if not os.path.isdir(input_dir):
        logging.error(f"Input directory '{input_dir}' is not a valid directory.")
        sys.exit(1)

    output_dir = f"{input_dir}/{output_dir}"
    create_dir(output_dir)

    images = natsorted(glob(f"{input_dir}/lines/*.jpg"))[:]
    labels = natsorted(glob(f"{input_dir}/transcriptions/*.txt"))

    logging.info(f"Images: {len(images)}, Labels:{len(labels)}")

    if len(images) == 0 or len(labels) == 0:
        logging.error("Have you provided all data?")
        sys.exit(1)

    characters = N_DEFAULT_CHARSET
    char_classes = len(characters) + 1

    if architecture == "CRNN":
        network = CRNNNetwork(num_classes=char_classes)
    else:
        network = EasterNetwork(num_classes=char_classes)

    ocr_trainer = OCRTrainer(network=network, image_paths=images, label_paths=labels)
    ocr_trainer.train(epochs=epochs)
