"""
A simple OCR Trainer class. 
- TODO:
    - add a metic (cer, wer, editdistance) for evaluation pass
    - add an interface to add one or multiple schedulers to the training loop (e.g. learning rate reduction or recude on plateau tracking)
    - add option for using mixed precision training (torch.cuda.amp.autocast)
    - add option to run one pass over the a test dataset

"""

import os
import json
import torch
import pyewts
import logging
from torch import nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from config import N_DEFAULT_CHARSET
from src.Datasets import CTCDataset, ctc_collate_fn
from src.Models import VanillaCRNN, Easter2
from src.Utils import create_dir, get_filename, shuffle_data, read_data


class Network:
    def __init__(self, model: nn.Module) -> None:
        self.device = "cuda"
        self.architecture = "ocr_architecture"
        self.image_height = 80
        self.image_width = 2000
        self.num_classes = len(N_DEFAULT_CHARSET) + 1
        self.model = model
        self.criterion = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def _get_device(self):
        # checks if the wanted device is actually available
        # add fallback to some other device
        return "cuda"

    def fine_tune(self, epochs: int):
        print(f"To be implemented")

    def forward(self, data):
        images, targets, target_lengths = [d.to(self.device) for d in data]

        logits = self.model(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        target_lengths = torch.flatten(target_lengths)

        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

        return loss

    def evaluate(self, data_loader, calculate_cer: bool = True):
        val_ctc_losses = []
        self.model.eval()

        for _, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, targets, target_lengths = [d.to(self.device) for d in data]
            with torch.no_grad():
                loss = self.forward(data)
                val_ctc_losses.append(loss / images.size(0))

        val_loss = torch.mean(torch.tensor(val_ctc_losses))

        return val_loss.item()

    def train(
        self,
        data_batch,
        clip_grads: bool = True,
        grad_clip: int = 5,
    ):
        self.model.train()

        loss = self.forward(data_batch)

        self.optimizer.zero_grad()
        loss.backward()

        if clip_grads:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        return loss.item()

    def get_checkpoint(self):
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        return checkpoint

    def get_config(self):
        print("returning the base config")

    def export_onnx(
        self, out_dir: str, model_name: str = "model", opset: int = 17
    ) -> None:
        self.model.eval()

        model_input = torch.randn(
            [self.num_classes, self.image_height, self.image_width], device=self.device
        )
        out_file = f"{out_dir}/{model_name}.onnx"
        torch.onnx.export(
            self.model,
            model_input,
            out_file,
            export_params=True,
            opset_version=opset,
            verbose=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        logging.info(f"Onnx file exported to: {out_file}")


class CRNNNetwork(Network):
    def __init__(
        self,
        name: str = "CRNN-Model",
        num_classes: int = 77,
        rnn_units: int = 256,
        hidden_units: int = 64,
    ) -> None:

        self.architecture = "crnn"
        self.name = (name,)
        self.image_width = 2000
        self.image_height = 80
        self.num_classes = num_classes
        self.hidden_units: int = hidden_units
        self.rnn_units: int = rnn_units
        self.rnn_type: str = "lstm"
        self.use_leaky_relu: bool = False
        self.learning_rate: float = 0.0005
        self.optimizer: str = "rmsprop"
        self.ctc_loss_reduction: str = "sum"
        self.device = "cuda"

        self.model = VanillaCRNN(
            img_height=self.image_height,
            img_width=self.image_width,
            charset_size=self.num_classes,
            map_to_seq_hidden=self.hidden_units,
            rnn_hidden=self.rnn_units,
            leaky_relu=self.use_leaky_relu,
            rnn=self.rnn_type,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.criterion = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)

        super().__init__(self.model)

    def fine_tune(self, epochs: int = 10):
        print("Fine tuning a CRNN...")
        super().fine_tune(epochs)

    def get_config(self):
        print("Returning crrn confing")
        super().get_config()


class EasterNetwork(Network):
    def __init__(self, name: str = "Easter2-Model", num_classes: int = 77) -> None:

        self.architecture = "easter2"
        self.name = name
        self.image_width = 2000
        self.image_height = 80
        self.num_classes = num_classes
        self.device = "cuda"
        self.learning_rate: float = 0.0005
        self.model = Easter2(
            input_width=self.image_width,
            input_height=self.image_height,
            vocab_size=self.num_classes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        #self.criterion = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
        self.criterion = CustomCTC()

        super().__init__(self.model)

    def forward(self, data):
        images, targets, target_lengths = data
        images = torch.squeeze(images).to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        logits = self.model(images)
        logits = logits.permute(2, 0, 1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor(
            [logits.size(0)] * batch_size
        )  # i.e. time steps
        target_lengths = torch.flatten(target_lengths)

        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

        return loss


class OCRTrainer:
    def __init__(
        self,
        network: Network,
        image_paths: list[str],
        label_paths: list[str],
        train_split: float = 0.8,
        val_test_split: float = 0.5,
        pre_load: bool = True,
        image_width: int = 2000,
        image_height: int = 80,
        max_label_length: int = 420,
        min_label_length: int = 30,
        charset: str = N_DEFAULT_CHARSET,
        batch_size: int = 32,
        workers: int = 4,
        output_dir: str = "Output",
        model_name: str = "OCRModel",
        do_test_pass: bool = True,
        calculate_cer: bool = True,
    ):
        self.network = network
        self.model_name = model_name
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.train_split = train_split
        self.val_test_split = val_test_split
        self.pre_load = pre_load
        self.image_width = image_width
        self.image_height = image_height
        self.max_label_length = max_label_length
        self.min_label_length = min_label_length

        self.batch_size = batch_size
        self.workers = workers
        self.charset = charset
        self.converter = pyewts.pyewts()
        self.do_test_pass = do_test_pass
        self.calculate_cer = calculate_cer
        self.training_time = datetime.now()

        self.output_dir = self._create_output_dir(output_dir)

        (
            self.train_images,
            self.train_labels,
            self.valid_images,
            self.valid_labels,
            self.test_images,
            self.test_labels,
        ) = self._init_datasets(self.train_split, self.val_test_split)

        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.charset = charset
        self.converter = pyewts.pyewts()

        self.train_dataset, self.valid_dataset, self.test_dataset = (
            self._build_datasets()
        )
        self.train_loader, self.valid_loader, self.test_loader = self._get_dataloaders()

    def _create_output_dir(self, output_dir) -> str:
        output_dir = os.path.join(
            output_dir,
            f"{self.training_time.year}_{self.training_time.month}_{self.training_time.day}_{self.training_time.hour}_{self.training_time.minute}",
        )
        create_dir(output_dir)
        return output_dir

    def _save_dataset(self):
        out_file = os.path.join(self.output_dir, "data.distribution")

        distribution = {}
        train_data = []
        valid_data = []
        test_data = []

        for sample in self.train_images:
            sample_name = get_filename(sample)
            train_data.append(sample_name)

        for sample in self.valid_images:
            sample_name = get_filename(sample)
            valid_data.append(sample_name)

        for sample in self.test_images:
            sample_name = get_filename(sample)
            test_data.append(sample_name)

        distribution["train"] = train_data
        distribution["validation"] = valid_data
        distribution["test"] = test_data

        with open(out_file, "w", encoding="UTF-8") as f:
            json.dump(distribution, f, ensure_ascii=False, indent=1)

        logging.info(f"Saved data distribution to: {out_file}.")

    def _init_datasets(self, train_split: float = 0.8, val_test_split: float = 0.5):
        images, labels = shuffle_data(self.image_paths, self.label_paths)

        _train_idx = int(len(images) * train_split)
        train_images = images[:_train_idx]
        train_labels = labels[:_train_idx]

        val_test_imgs = images[_train_idx:]
        val_test_lbls = labels[_train_idx:]

        _val_idx = int(len(val_test_imgs) * val_test_split)

        valid_images = val_test_imgs[:_val_idx]
        valid_labels = val_test_lbls[:_val_idx]

        test_images = val_test_imgs[_val_idx:]
        test_labels = val_test_lbls[_val_idx:]

        logging.info(
            f"Train Images: {len(train_images)}, Train Labels: {len(train_labels)}"
        )
        logging.info(
            f"Validation Images: {len(valid_images)}, Validation Images: {len(valid_labels)}"
        )

        logging.info(
            f"Test Images: {len(test_images)}, Test Labels: {len(test_labels)}"
        )

        return (
            train_images,
            train_labels,
            valid_images,
            valid_labels,
            test_images,
            test_labels,
        )

    def _build_datasets(self):

        self._save_dataset()

        train_images, train_labels = read_data(
            self.train_images,
            self.train_labels,
            self.converter,
            min_label_length=self.min_label_length,
            max_label_length=self.max_label_length,
        )
        valid_images, valid_labels = read_data(
            self.valid_images,
            self.valid_labels,
            self.converter,
            min_label_length=self.min_label_length,
            max_label_length=self.max_label_length,
        )
        test_images, test_labels = read_data(
            self.test_images,
            self.test_labels,
            self.converter,
            min_label_length=self.min_label_length,
            max_label_length=self.max_label_length,
        )

        train_dataset = CTCDataset(
            images=train_images,
            labels=train_labels,
            img_height=self.image_height,
            img_width=self.image_width,
            charset=self.charset,
        )
        valid_dataset = CTCDataset(
            images=valid_images,
            labels=valid_labels,
            img_height=self.image_height,
            img_width=self.image_width,
            charset=self.charset,
        )

        test_dataset = CTCDataset(
            images=test_images,
            labels=test_labels,
            img_height=self.image_height,
            img_width=self.image_width,
            charset=self.charset,
        )

        return train_dataset, valid_dataset, test_dataset

    def _get_dataloaders(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ctc_collate_fn,
            drop_last=True,
            num_workers=self.workers,
            persistent_workers=True,
        )

        valid_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ctc_collate_fn,
            drop_last=True,
            num_workers=self.workers,
            persistent_workers=True,
        )

        test_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ctc_collate_fn,
            drop_last=True,
            num_workers=self.workers,
            persistent_workers=True,
        )

        try:
            next(iter(train_loader))
            next(iter(valid_loader))
            next(iter(test_loader))

        except BaseException as e:
            logging.error(f"Failed to iterate over dataset: {e}.")

        return train_loader, valid_loader, test_loader

    def _save_checkpoint(self, save_path: str):
        chpt_file = f"{save_path}.pth"
        checkpoint = self.network.get_checkpoint()
        torch.save(checkpoint, chpt_file)
        logging.info(f"Saved checkpoint to: {chpt_file}.")

    def _save_history(self, history: dict):
        out_file = os.path.join(self.output_dir, "history.txt")

        with open(out_file, "w", encoding="UTF-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=1)

        logging.info(f"Training history saved to: {out_file}.")

    def _save_model_config(self, save_path: str):
        _charset = self.charset
        out_file = os.path.join(self.output_dir, "model_config.json")

        network_config = {
            "model": save_path,
            "architecture": self.network.architecture,
            "input_width": self.image_width,
            "input_height": self.image_height,
            "charset": _charset,
        }
        json_out = json.dumps(network_config, ensure_ascii=False, indent=2)

        with open(out_file, "w", encoding="UTF-8") as f:
            f.write(json_out)

        logging.info(f"Saved model config to: {out_file}")

    def train(self, epochs: int = 10):
        logging.info(f"Training network....")
        save_path = os.path.join(self.output_dir, self.model_name)

        train_history = {}
        train_loss_history = []
        val_loss_history = []
        best_loss = None

        for epoch in range(1, epochs + 1):
            epoch_train_loss = 0
            tot_train_count = 0

            for _, data in tqdm(
                enumerate(self.train_loader), total=len(self.train_loader)
            ):
                # TODO: the data unpacking happens twice, refactor this
                images, targets, target_lengths = data
                train_loss = self.network.train(data)
                train_size = images.size(0)

                epoch_train_loss += train_loss
                tot_train_count += train_size

            train_loss = epoch_train_loss / tot_train_count
            logging.info(f"Epoch {epoch} => Train-Loss: {train_loss}")
            train_loss_history.append(train_loss)

            val_loss = self.network.evaluate(self.valid_loader)
            logging.info(f"Epoch {epoch} => Val-Loss: {val_loss}")
            val_loss_history.append(val_loss)

            if best_loss is None:
                best_loss = val_loss
                self._save_checkpoint(save_path)

            elif val_loss < best_loss:
                self._save_checkpoint(save_path)

        train_history["train_losses"] = train_loss_history
        train_history["val_losses"] = val_loss_history

        self._save_history(train_history)
        self._save_model_config(save_path)
        self.network.export_onnx(self.output_dir)

        logging.info(f"Training complete.")


class CustomCTC(nn.Module):
    def __init__(self, gamma: float = 0.5, alpha: int = 0.25, blank: int = 0):
        super(CustomCTC, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.blank = blank

    def forward(self, log_probs, labels, input_lengths, target_lenghts):
        ctc_loss = nn.CTCLoss(blank=self.blank, reduction='sum', zero_infinity=True)(log_probs, labels, input_lengths,
                                                                                     target_lenghts)
        p = torch.exp(-ctc_loss)
        loss = self.alpha * (torch.pow((1 - p), self.gamma)) * ctc_loss

        return loss
