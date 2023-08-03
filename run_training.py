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
from config import DEFAULT_CHARSET
from src.Modules import OCRTrainer
from src.Utils import create_dir




if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--charset", type=str, required=False)

    args = parser.parse_args()

    input_dir = args.input_dir

    if not os.path.isdir:
        logging.error(f"Input directory '{input_dir}' is not a valid directory.")
        sys.exit(1)

    if not args.output_dir:
        output_dir = os.path.join(input_dir, "Output")
        create_dir(output_dir)
    else:
        output_dir = args.output_dir
        create_dir(output_dir)

    images = natsorted(glob(f"{input_dir}/lines/*.jpg"))
    labels = natsorted(glob(f"{input_dir}/transcriptions/*.txt"))

    logging.info(f"Images: {len(images)}, Labels:{len(labels)}")

    if len(images) == 0 or len(labels) == 0:
        logging.error("Have you provided all data?")
        sys.exit(1)

    ocr_trainer = OCRTrainer(images, labels, output_dir=output_dir)
    ocr_trainer.train()


    