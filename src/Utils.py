import os
import re
import cv2
import json
import logging
import random
import numpy as np
from tqdm import tqdm


def create_dir(dir_path: str) -> None:
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except BaseException as e:
        logging.error(f"Failed to create directory: {e}")


def get_filename(x):
    return os.path.basename(x).split(".")[0]


def validate_data(images_paths: list[str], label_paths: list[str]):
    """
    Attempts to remove small and empty files, builds a common list if images and labels if there is a mismatch between both
    """
    # filters empty text files and white pages
    images = [x for x in images_paths if os.stat(x).st_size >= 3000]
    labels = [x for x in label_paths if os.stat(x).st_size != 0]

    image_list = list(map(get_filename, images))
    transcriptions_list = list(map(get_filename, labels))

    return list(set(image_list) & set(transcriptions_list))


def get_train_data(summary_file: str, dataset_dir: str) -> tuple[list[str], list[str]]:
    print(dataset_dir)
    f = open(summary_file, "r")
    content = f.read()
    json_data = json.loads(content)

    train_images_file = json_data["train_images"]
    valid_images_file = json_data["valid_images"]

    f = open(train_images_file, "r")
    train_image_names = f.readlines()
    train_image_names = [x.strip() for x in train_image_names]
    train_images = [f"{dataset_dir}/lines/{x}.jpg" for x in train_image_names]
    train_labels = [f"{dataset_dir}/transcriptions/{x}.txt" for x in train_image_names]

    f = open(valid_images_file, "r")
    valid_image_names = f.readlines()
    valid_image_names = [x.strip() for x in valid_image_names]
    valid_images = [f"{dataset_dir}/lines/{x}.jpg" for x in valid_image_names]
    valid_labels = [f"{dataset_dir}/transcriptions/{x}.txt" for x in valid_image_names]

    return train_images, train_labels, valid_images, valid_labels


def shuffle_data(images, labels):
    c = list(zip(images, labels))
    random.shuffle(c)

    a, b = zip(*c)

    return list(a), list(b)


def resize_to_height(image, target_height: int):
    width_ratio = target_height / image.shape[0]
    image = cv2.resize(image, (int(image.shape[1] * width_ratio), target_height))
    return image


def resize_to_width(image, target_width: int):
    width_ratio = target_width / image.shape[1]
    image = cv2.resize(image, (target_width, int(image.shape[0] * width_ratio)))
    return image


def resize(img: np.array, target_width: int, target_height: int):
    width_ratio = target_width / img.shape[1]
    height_ratio = target_height / img.shape[0]

    if width_ratio < height_ratio:  # maybe handle equality separately
        tmp_img = resize_to_width(img, target_width)

    elif width_ratio > height_ratio:
        tmp_img = resize_to_height(img, target_height)

    else:
        tmp_img = resize_to_width(img, target_width)

    return cv2.resize(tmp_img, (target_width, target_height))


def resize_n_pad(
    img: np.array, target_width: int, target_height: int, padding: str
) -> np.array:
    """
    Preliminary implementation of resizing and padding images.
    Args:
        - padding: "white" for padding the image with 255, otherwise the image will be padded with 0

    - TODO: using np.pad for an eventually more elegant/faster implementation
    """
    width_ratio = target_width / img.shape[1]
    height_ratio = target_height / img.shape[0]

    if width_ratio < height_ratio:  # maybe handle equality separately
        tmp_img = resize_to_width(img, target_width)

        if padding == "white":
            v_stack = np.ones(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        else:
            v_stack = np.zeros(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        v_stack *= 255

        out_img = np.vstack([v_stack, tmp_img])

    elif width_ratio > height_ratio:
        tmp_img = resize_to_height(img, target_height)

        if padding == "white":
            h_stack = np.ones(
                (tmp_img.shape[0], target_width - tmp_img.shape[1]), dtype=np.uint8
            )
        else:
            h_stack = np.zeros(
                (tmp_img.shape[0], target_width - tmp_img.shape[1]), dtype=np.uint8
            )
        h_stack *= 255

        out_img = np.hstack([tmp_img, h_stack])
    else:
        tmp_img = resize_to_width(img, target_width)

        if padding == "white":
            v_stack = np.ones(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        else:
            v_stack = np.zeros(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        v_stack *= 255

        out_img = np.vstack([v_stack, tmp_img])
        # logging.info(f"Info -> equal ratio: {img.shape}, w_ratio: {width_ratio}, h_ratio: {height_ratio}")

    return cv2.resize(out_img, (target_width, target_height))


def clean_unicode_label(l, full_bracket_removal: bool = True):
    """
    Some preliminary clean-up rules for the Unicode text.
    - Note: () are just removed. This was valid in case of the Lhasa Kanjur.
    In other e-texts, a complete removal of the round and/or square brackets together with the enclosed text should be applied
    in order to remove interpolations, remarks or similar additions.
    In such cases set full_bracket_removal to True.
    """

    l = re.sub("[\uf8f0]", " ", l)
    l = re.sub("[\xa0]", "", l)
    l = re.sub("[༌]", "་", l)  # replace triangle tsheg with regular

    if full_bracket_removal:
        l = re.sub("[\[(].*?[\])]", "", l)
    else:
        l = re.sub("[()]", "", l)
    return l


def preprocess_wylie(line: str) -> str:
    line = line.replace("/ /", "/_/")
    line = line.replace("/ ", "/")

    return line


def post_process_wylie(line: str) -> str:
    line = line.replace("\\u0f85", "&")
    line = line.replace("\\u0f09", "ä")
    line = line.replace("\\u0f13", "ö")
    line = line.replace("\\u0f12", "ü")
    line = line.replace("  ", " ")
    line = line.replace("_", "")
    line = line.replace(" ", "§")
    line = re.sub("[\[(].*?[\])]", "", line)
    return line


def read_data(
    image_list,
    label_list: list,
    converter,
    min_label_length: int = 30,
    max_label_length: int = 240,
) -> tuple[list[str], list[str]]:
    """
    Reads all labels into memory, filter labels for min_label_length and max_label_length.
    # TODO:
    1) convert the training labels to wylie ahead of training and clean them up avoiding multiple checks while reading the dataset

    """
    labels = []
    images = []
    for image_path, label_path in tqdm(
        zip(image_list, label_list), total=len(label_list), desc="reading labels"
    ):
        f = open(label_path, "r", encoding="utf-8")
        label = f.readline()
        label = clean_unicode_label(label)

        if min_label_length < len(label) < max_label_length:
            label = converter.toWylie(label)
            label = post_process_wylie(label)

            if not "\\u" in label:  # filter out improperly converted unicode signs
                labels.append(label)
                images.append(image_path)

    return images, labels
