"""
A simple interface for fine-tuning a pre-trained OCR network. 
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
    parser.add_argument("--epochs", type=int, required=False, default=20)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--optimizer", choices=["rmsprop", "adam"], required=False, default="adam")
    parser.add_argument("--lr", type=float, required=False, default=0.0005)
    parser.add_argument("--train_val_split", type=float, required=False, default=0.2)
    parser.add_argument("--max_seq_length", type=int, required=False, default=302)
    parser.add_argument("--min_seq_length", type=int, required=False, default=30)
    args = parser.parse_args()

    input_dir = args.input_dir
    model_checkpoint = args.model_checkpoint
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    train_val_split = args.train_val_split
    max_seq = args.max_seq_length
    min_seq = args.min_seq_length
    optimizer = str(args.optimizer)

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

    ocr_trainer = OCRTrainer(
        images, labels, 
        output_dir=output_dir,
        batch_size=batch_size,
        max_label_length=max_seq,
        min_label_length=min_seq,
        train_val_split_ratio=train_val_split)
    ocr_trainer.train(model_checkpoint=model_checkpoint, epochs=epochs, optimizer=optimizer, learning_rate=learning_rate)