
import cv2
import torch
import pyewts
from typing import Optional
from torch.utils.data import Dataset
from src.Utils import resize_n_pad
from config import DEFAULT_CHARSET


class CTCDataset(Dataset):
    def __init__(
        self,
        images: list,
        labels: list,
        charset: str = DEFAULT_CHARSET,
        img_height: int = 80,
        img_width: int = 2000,
        augmentations: Optional[list[str]] = None,
    ):
        super(CTCDataset, self).__init__()

        self.images = images
        self.labels = labels
        self.img_height = img_height
        self.img_width = img_width
        self.charset = charset
        self.encode_label = {char: i + 1 for i, char in enumerate(self.charset)}
        self.decode_label = {label: char for char, label in self.encode_label.items()}
        self.converter = pyewts.pyewts()
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])  # grayscale

        if self.augmentations is not None:
            aug = self.augmentations(image=image)

            image = aug["image"]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = resize_n_pad(
            image,
            target_width=self.img_width,
            target_height=self.img_height,
            padding="black",
        )
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        target = [self.encode_label[c] for c in self.labels[index]]
        target_length = [len(target)]

        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)

        return image, target, target_length


def ctc_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
