"""
A simple OCR Trainer class. 
- TODO:
    - add a metic (cer, wer, editdistance) for evaluation pass
    - add an interface to add one or multiple schedulers to the training loop (e.g. learning rate reduction or recude on plateau tracking)
    - add option for using mixed precision training (torch.cuda.amp.autocast)
    - add option to run one pass over the a test dataset

"""
import os
import sys
import torch
import pyewts
import logging
from tqdm import tqdm
from datetime import datetime
from torch.nn import CTCLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import DEFAULT_CHARSET
from src.Datasets import CTCDataset, ctc_collate_fn
from src.Models import VanillaCRNN
from src.Utils import create_dir, shuffle_data, read_data


class OCRTrainer:
    def __init__(
        self,
        image_paths: list[str],
        label_paths: list[str],
        train_val_split: float = 0.8,
        image_width: int = 2000,
        image_height: int = 80,
        batch_size: int = 32,
        output_dir: str = "Output",
        charset: str = DEFAULT_CHARSET,
        max_label_length: int = 240,
        min_label_length: int = 30
    ):
        self.model = None
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.max_label_length = max_label_length
        self.min_label_length = min_label_length
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.training_time = datetime.now()
        self.output_dir = self.create_output_dir(output_dir)

        (
            self.train_images,
            self.train_labels,
            self.valid_images,
            self.valid_labels,
        ) = self._init_datasets(self.train_val_split)

        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.charset = charset
        self.converter = pyewts.pyewts()

        self.train_dataset, self.valid_dataset = self._build_datasets()
        self.train_loader, self.valid_loader = self._get_dataloaders()


    def create_output_dir(self, output_dir) -> str:
        output_dir = os.path.join(
            output_dir,
            f"{self.training_time.year}_{self.training_time.month}_{self.training_time.day}_{self.training_time.hour}_{self.training_time.minute}",
        )
        create_dir(output_dir)
        return output_dir

    def _init_datasets(self, split_ratio: float = 0.8):
        images, labels = shuffle_data(self.image_paths, self.label_paths)
        train_images = images[:int(len(images)*split_ratio)]
        train_labels = labels[:int(len(labels)*split_ratio)]

        valid_images = images[int(len(images)*split_ratio):]
        valid_labels = labels[int(len(labels)*split_ratio):]

        logging.info(
            f"Train Images: {len(train_images)}, Train Labels: {len(train_labels)}"
        )
        logging.info(
            f"Valid Images: {len(valid_images)}, Valid Images: {len(valid_labels)}"
        )

        return train_images, train_labels, valid_images, valid_labels

    def _save_dataset(self):
        # save train data
        train_images_out = f"{self.output_dir}/train_images.txt"
        train_labels_out = f"{self.output_dir}/train_labels.txt"

        with open(train_images_out, "w") as f:
            for img in self.train_images:
                f.write(f"{img}\n")

        with open(train_labels_out, "w") as f:
            for lbl in self.train_labels:
                f.write(f"{lbl}\n")

        # save validation data
        val_images_out = f"{self.output_dir}/val_images.txt"
        val_labels_out = f"{self.output_dir}/val_labels.txt"

        with open(val_images_out, "w") as f:
            for img in self.valid_images:
                f.write(f"{img}\n")

        with open(val_labels_out, "w") as f:
            for lbl in self.valid_labels:
                f.write(f"{lbl}\n")

    def _build_datasets(self):
        # saving the dataset before reading the actual labels
        self._save_dataset()

        self.train_images, self.train_labels = read_data(
            self.train_images, self.train_labels, self.converter, min_label_length=self.min_label_length, max_label_length=self.max_label_length
        )
        self.valid_images, self.valid_labels = read_data(
            self.valid_images, self.valid_labels, self.converter, min_label_length=self.min_label_length, max_label_length=self.max_label_length
        )

        train_dataset = CTCDataset(
            images=self.train_images,
            labels=self.train_labels,
            img_height=self.image_height,
            img_width=self.image_width,
            charset=self.charset,
        )
        valid_dataset = CTCDataset(
            images=self.valid_images,
            labels=self.valid_labels,
            img_height=self.image_height,
            img_width=self.image_width,
            charset=self.charset,
        )

        return train_dataset, valid_dataset

    def _get_dataloaders(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ctc_collate_fn,
            drop_last=True,
            num_workers=4,
            persistent_workers=True
        )

        valid_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ctc_collate_fn,
            drop_last=True,
            num_workers=4,
            persistent_workers=True
        )

        try:
            next(iter(train_loader))
            next(iter(valid_loader))

        except BaseException as e:
            logging.error(f"Failed to iterate over Dataset: {e}")

        return train_loader, valid_loader

    def _save_checkpoint(self, network, optimizer, save_path):
        checkpoint = {
            "state_dict": network.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)
        logging.info(f"Saved Model Checkpoint at {save_path}")

    def _check_accuracy(
        self, network, data_loader, device: torch.device, criterion: CTCLoss
    ):
        val_ctc_losses = []
        network.eval()

        for _, val_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            with torch.no_grad():
                images, targets, target_lengths = [d.to(device) for d in val_data]

                logits = network(images)
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)

                batch_size = images.size(0)
                input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
                target_lengths = torch.flatten(target_lengths)

                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_ctc_losses.append(loss / batch_size)

        val_loss = torch.mean(torch.tensor(val_ctc_losses))
        network.train()

        return val_loss.item()

    def _train_batch(
        self,
        network,
        batch_data,
        optimizer,
        criterion,
        device,
        clip_grads: bool = True,
        grad_clip: int = 5,
    ):
        network.train()
        images, targets, label_lengths = [data.to(device) for data in batch_data]

        logits = network(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        label_lengths = torch.flatten(label_lengths)

        loss = criterion(log_probs, targets, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()

        if clip_grads:
            torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
        optimizer.step()

        return loss.item()

    def _train_network(
        self,
        network,
        device,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        checkpoint_path: str,
        epochs: int = 10,
    ) -> tuple[list, list]:

        train_loss_history = []
        val_loss_history = []
        best_loss = None

        for epoch in range(1, epochs + 1):
            epoch_train_loss = 0
            tot_train_count = 0

            for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):

                loss = self._train_batch(
                    network, data, optimizer, criterion, device
                )
                train_size = data[0].size(0)

                epoch_train_loss += loss
                tot_train_count += train_size

            train_loss = epoch_train_loss / tot_train_count
            train_loss_history.append(train_loss)

            val_loss = self._check_accuracy(network, val_loader, device, criterion)
            val_loss_history.append(val_loss)

            logging.info(
                f"Epoch {epoch} => train_loss: {train_loss}, val loss: {val_loss}"
            )

            if best_loss is None:
                best_loss = val_loss
                self._save_checkpoint(network, optimizer, checkpoint_path)

            elif val_loss < best_loss:
                self._save_checkpoint(network, optimizer, checkpoint_path)

        return train_loss_history, val_loss_history
    

    def _save_train_data(self, network, train_losses: list, val_losses: list) -> None:
        train_losses_file = f"{self.output_dir}/train_losses.txt"
        val_losses_file = f"{self.output_dir}/val_losses.txt"
        network_info_file = f"{self.output_dir}/val_losses.txt"

        with open(train_losses_file, "w") as f:
            f.write(str(train_losses))

        with open(val_losses_file, "w") as f:
            f.write(str(val_losses))

        with open(network_info_file, "w") as f:
            f.write(str(network))

        logging.info("Saved train stats to '{self.output_dir}.")

    def train(
        self,
        network_name: str = "crnn_network",
        epochs: int = 10,
        hidden_units: int = 64,
        rnn_units: int = 256,
        rnn_type: str = "lstm",
        use_leaky_relu: bool = False,
        learning_rate: float = 0.0005,
        optimizer: str = "rmsprop",
        ctc_loss_reduction: str = "sum",
        model_checkpoint: str = None
    ):
        torch.cuda.empty_cache()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        network = VanillaCRNN(
            img_height=self.image_height,
            img_width=self.image_width,
            charset_size=len(self.charset)+1,
            map_to_seq_hidden=hidden_units,
            rnn_hidden=rnn_units,
            leaky_relu=use_leaky_relu,
            rnn=rnn_type,
        )

        if model_checkpoint is not None:
            try:
                logging.info(f"Loading checkpoint: {model_checkpoint}")
                loaded_checkpt = torch.load(model_checkpoint)
                network.load_state_dict(loaded_checkpt['state_dict'])
                
                for param in network.parameters():
                    param.requires_grad = False


                for param in network.dense.parameters():
                    param.requires_grad = True

                for param in network.rnn2.parameters():
                    param.requires_grad = True

                for param in network.rnn1.parameters():
                    param.requires_grad = True

                for param in network.linear.parameters():
                    param.requires_grad = True

                for param in network.conv_block_6.parameters():
                    param.requires_grad = True

                logging.info("Successfully loaded provided checkpoint. Fine tuning model...")


            except BaseException as e:
                logging.info(f"Failed to load model checkpoint, training from scratch: {e}")
                sys.exit(1)

        network.to(device)

        if optimizer == "rmsprop":
            optimizer = RMSprop(network.parameters(), lr=learning_rate, centered=True)
        elif optimizer == "adam":
            optimizer = Adam(network.parameters(), lr=learning_rate)
        else:
            logging.info(
                "No valid argument for the optimizer was provided, please use either 'rmsprop' or 'adam'."
            )


        criterion = CTCLoss(reduction=ctc_loss_reduction, zero_infinity=True)
        criterion.to(device)

        model_save_path = f"{self.output_dir}/{network_name}.pth"
        train_losses, val_losses = self._train_network(
            network,
            device,
            optimizer,
            criterion,
            self.train_loader,
            self.valid_loader,
            model_save_path,
            epochs=epochs,
        )

        self._save_train_data(network, train_losses, val_losses)

        logging.info("Training is finished :)")