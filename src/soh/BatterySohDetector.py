import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from BatterySohDataset import BatteryDataset
from LSTMSohModel import LSTMSOHModel
from src.config.Config import Config
import time

from src.soh.FnnSohModel import FNNSOHModel


class BatterySOHDetector:
    def __init__(self, model, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.config = config
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)
        self.losses = []

    def train(self):
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            epoch_loss = 0
            start_time = time.time()

            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            self.losses.append(avg_loss)
            epoch_duration = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            print(
                f"Epoch [{epoch + 1}/{self.config.EPOCHS}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}, Time: {epoch_duration:.2f}s")

            self.scheduler.step()

        self.plot_losses()

    def evaluate(self):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                predictions = self.model(X_batch)
                y_pred.extend(predictions.numpy().flatten())
                y_true.extend(y_batch.numpy().flatten())

        mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
        print(f"MSE: {mse:.4f}")
        # Plot y_true vs y_pred
        plt.figure()
        plt.plot(y_true, label='Actual SOH', marker='o', linestyle='dashed')
        plt.plot(y_pred, label='Predicted SOH', marker='x')
        plt.xlabel("Samples")
        plt.ylabel("State of Health (SOH)")
        plt.title("Actual vs Predicted SOH")
        plt.legend()
        plt.grid()
        # plt.show()
        save_path = os.path.join(self.config.FIGURE_SAVE_PATH, "soh_eval_result.png")
        plt.savefig(save_path)
        # plt.show()
        print(f"Figure saved to {save_path}")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32)).numpy()

    def save_model(self, path="battery_soh_model.pth"):
        save_path = os.path.join(self.config.MODEL_SAVE_PATH, path)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {self.config.MODEL_SAVE_PATH}")

    def plot_losses(self, path="soh_training_loss.png"):
        plt.figure()
        plt.plot(self.losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        # Combine directory path and figure name
        save_path = os.path.join(self.config.FIGURE_SAVE_PATH, path)
        plt.savefig(save_path)
        # plt.show()
        print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    file_paths = "../../data/soh/all.csv"
    dataset = BatteryDataset(file_paths, 0.2, 0.9)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    eval_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    input_size = train_dataset.dataset.X.size(2)

    model_lstm = LSTMSOHModel(input_size=input_size, hidden_size=64, num_layers=2)
    # model = FNNSOHModel(input_size=input_size, hidden_size=64, num_layers=3)
    detector = BatterySOHDetector(model_lstm, train_loader, eval_loader, Config)
    detector.train()
    detector.evaluate()
    detector.save_model()
