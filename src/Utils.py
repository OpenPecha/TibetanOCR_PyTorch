import os
import re
import cv2
import json
import logging
import random
import numpy as np
from tqdm import tqdm
from enum import Enum


class Labelformat(Enum):
    t_unicode = 0
    wylie = 1


def create_dir(dir_path: str) -> None:
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Created output directory: {dir_path}")
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


def read_distribution(file_path: str):
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            content = json.loads(content)

            if "train" in content and "validation" in content and "test" in content:
                train_samples = content["train"]
                valid_samples = content["validation"]
                test_samples = content["test"]
                return train_samples, valid_samples, test_samples
            else:
                logging.error(f"Data distribution is missing the required keys 'train' and 'validation' and 'test'.")
                return None, None, None

    else:
        logging.error(f"Specified distribution file does not exist: {file_path}")
        return None, None, None

def split_dataset(
    image_paths: list[str],
    label_paths: list[str],
    batch_size: int = 32,
    even_splits: bool = True,
    train_val_split: float = 0.8,
    val_test_split: float = 0.5,
):
    """
    Splitting the dataset evenly for the selected batch size.
    """

    batches = len(image_paths) // batch_size
    train_batches = int(batches * train_val_split)

    train_images = image_paths[: train_batches * batch_size]
    train_labels = label_paths[: train_batches * batch_size]

    assert len(train_images) % batch_size == 0 and len(train_labels) % batch_size == 0

    val_test_images = image_paths[train_batches * batch_size :]
    val_test_labels = label_paths[train_batches * batch_size :]
    val_test_split = int(
        ((len(val_test_images) * val_test_split) // batch_size) * batch_size
    )

    val_images = val_test_images[:val_test_split]
    val_labels = val_test_labels[:val_test_split]

    test_images = val_test_images[val_test_split:]
    test_labels = val_test_labels[val_test_split:]

    if even_splits:
        test_images = test_images[: (len(test_images) // batch_size) * batch_size]
        test_labels = test_labels[: (len(test_images) // batch_size) * batch_size]

    print(f"Train Images: {len(train_images)}, Train Labels: {len(train_labels)}")
    print(f"Val Images: {len(val_images)}, Val Labels: {len(val_labels)}")
    print(f"Test Images: {len(test_images)}, Test Labels: {len(test_labels)}")

    assert len(train_images) % batch_size == 0 and len(train_labels) % batch_size == 0
    assert len(val_images) % batch_size == 0 and len(val_labels) % batch_size == 0
    assert len(test_images) % batch_size == 0 and len(test_labels) % batch_size == 0

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


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


def preprocess_unicode(l, full_bracket_removal: bool = True):
    """
    Some preliminary clean-up rules for the Unicode text.
    - Note: () are just removed. This was valid in case of the Lhasa Kanjur.
    In other e-texts, a complete removal of the round and/or square brackets together with the enclosed text should be applied
    in order to remove interpolations, remarks or similar additions.
    In such cases set full_bracket_removal to True.
    """

    l = l.replace("\uf8f0", " ")
    l = l.replace("\xa0", "")
    l = l.replace("\u200d", "")
    l = l.replace("༌", "་")  # replace triangle tsheg with regular

    if full_bracket_removal:
        l = re.sub("[\[(].*?[\])]", "", l)
    else:
        l = re.sub("[()]", "", l)
    return l


def preprocess_wylie_label(label: str) -> str:
    label = label.replace("༈", "!")
    label = label.replace("༅", "#")
    label = label.replace("|", "/")  # TODO: let sb. verify this choice is ok
    label = label.replace("/ /", "/_/")
    label = label.replace("/ ", "/")

    return label


def postprocess_wylie_label(label: str) -> str:
    label = label.replace("\\u0f85", "&")
    label = label.replace("\\u0f09", "ä")
    label = label.replace("\\u0f13", "ö")
    label = label.replace("\\u0f12", "ü")
    label = label.replace("\\u0fd3", "@")
    label = label.replace("\\u0fd4", "#")
    label = label.replace("\\u0f00", "oM")
    label = label.replace("\\u0f7f", "}")
    label = label.replace("*", " ")
    label = label.replace("  ", " ")
    label = label.replace("_", "")
    label = label.replace(" ", "§")  # specific encoding for the tsheg

    label = re.sub("[\[(].*?[\])]", "", label)
    return label


def read_data(
    image_list,
    label_list: list,
    converter,
    min_label_length: int = 30,
    max_label_length: int = 320,
    format: Labelformat = Labelformat.t_unicode,
) -> tuple[list[str], list[str]]:
    """
    Reads all labels into memory(!), filter labels for min_label_length and max_label_length.
    """
    labels = []
    images = []
    for image_path, label_path in tqdm(
        zip(image_list, label_list), total=len(label_list), desc="reading labels"
    ):
        f = open(label_path, "r", encoding="utf-8")
        label = f.readline()

        if format == Labelformat.t_unicode:
            try:
                label = preprocess_unicode(label)
            except BaseException as e:
                print(f"Failed to preprocess unicode label: {label_path}, {e}")

            if min_label_length < len(label) < max_label_length:
                label = converter.toWylie(label)
                label = postprocess_wylie_label(label)

                if not "\\u" in label:  # filter out improperly converted unicode signs
                    labels.append(label)
                    images.append(image_path)
        else:
            label = preprocess_wylie_label(label)
            label = postprocess_wylie_label(label)

            if min_label_length < len(label) < max_label_length:
                labels.append(label)
                images.append(image_path)

    return images, labels
