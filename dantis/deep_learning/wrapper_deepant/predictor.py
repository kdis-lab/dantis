import numpy as np
import torch
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from .early_stopping import EarlyStopping
from .model import DeepAnTCNN

import os

def retrieve_save_path(save_path, default_file_name):
    file_name = default_file_name
    # splits path into directories without file name and file name only
    file_path_tuple = os.path.split(save_path)
    if file_path_tuple[1] != "":
        file_name = file_path_tuple[1]
    if not os.path.exists(file_path_tuple[0]):
        os.mkdir(file_path_tuple[0])
    save_name = os.path.join(file_path_tuple[0], file_name)
    return save_name


class Predictor():
    def __init__(self, window, pred_window, lr = 1e-5, batch_size = 45, in_channels=1, filter1_size = 128, filter2_size = 32,
            kernel_size = 2, pool_size = 2, stride = 1):
        self.model = DeepAnTCNN(window, pred_window, in_channels, filter1_size, filter2_size, kernel_size, pool_size, stride)
        self.lr = lr
        self.batch_size = batch_size

    def train(self, train_dataset: Dataset, valid_dataset: Dataset, n_epochs, save_path, log_freq=10, early_stopping_patience = 5, early_stopping_delta = 1e-2):
        model_save_name = retrieve_save_path(save_path, "model.pt")

        valid_loss_min = np.Inf
        train_loss_min = np.Inf

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        criterion = L1Loss()

        dataloader_train = DataLoader(train_dataset, batch_size=self.batch_size)
        dataloader_valid = DataLoader(valid_dataset, batch_size=self.batch_size)

        early_stopping = EarlyStopping(early_stopping_patience, early_stopping_delta, n_epochs)

        for epoch in early_stopping:
            train_losses = []
            for X, y in dataloader_train:
                # training
                self.model.train()

                optimizer.zero_grad()
                output = self.model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = sum(train_losses)

            valid_losses = []
            for X, y in dataloader_valid:
                # validation
                self.model.eval()
                output_valid = self.model(X)

                loss_valid = criterion(output_valid, y)
                valid_losses.append(loss_valid.item())
            valid_loss = sum(valid_losses)

            early_stopping.update(valid_loss)
            if(epoch % log_freq == 0):
                print(f"Epoch: {epoch}/{n_epochs} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}")

            if train_loss < train_loss_min:
                train_loss_min = train_loss

            # save model if validation loss decreases
            if valid_loss < valid_loss_min:
                torch.save(self.model.state_dict(), model_save_name)
                valid_loss_min = valid_loss
            last_epoch = epoch
        if last_epoch < n_epochs:
            print(f"\nTraining canceled because validation loss has not decreased significantly for {early_stopping_patience} epochs")
            print(f"Minimal Training Loss: {train_loss_min:.6f} \tMinimal Validation Loss: {valid_loss_min:.6f}")
            print("Model has been saved")

    def predict(self, test_dataset: Dataset):
        self.model.eval()

        dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
        result = []
        for x, _ in dataloader:
            out = self.model(x).detach()
            result.append(out)
        return torch.cat(result, dim=0)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
