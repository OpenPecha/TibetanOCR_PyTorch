"""
A simple interface for fine-tuning a pre-trained OCR network. 

use e.g. python fine_tune.py --input_dir "D:/Datasets/Tibetan/Glomanthang/Annotations/GlomanThang_Dataset_October2023" --model_checkpoint "Checkpoints/2023_9_13_7_59/LhasaKanjur_prodigy_v3.pth"
"""

import os
import sys
import logging
import argparse
from glob import glob
from natsort import natsorted
from src.Modules import OCRTrainer
from src.Utils import create_dir



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--model_checkpoint", type=str, required=False)

    args = parser.parse_args()

    input_dir = args.input_dir
    model_checkpoint = args.model_checkpoint

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

    assert(len(images) == len(labels))

    logging.info(f"Images: {len(images)}, Labels:{len(labels)}")

    if len(images) == 0 or len(labels) == 0:
        logging.error("Have you provided all data?")
        sys.exit(1)

    ocr_trainer = OCRTrainer(images, labels, output_dir=output_dir)
    ocr_trainer.train(model_checkpoint=model_checkpoint)