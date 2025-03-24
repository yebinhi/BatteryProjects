import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

from src.config.Config import Config
from src.soc.BatterySOCDataset import BatterySOCDataset
from src.soc.LSTMModel import LSTMModel
from src.soc.TtlModel import TTLTransformerBlock


class BatterySOCDetector:
    def __init__(self, model, config=Config):
        self.device = config.DEVICE
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)
        self.config = config
        print(f"Using device: {self.device}")

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.config.EPOCHS):
            epoch_loss = 0
            start_time = time.time()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_duration = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            avg_loss = epoch_loss / len(train_loader)
            print(
                f"Epoch [{epoch + 1}/{self.config.EPOCHS}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}, Time: {epoch_duration:.2f}s")
            self.scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        return predictions

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        return {'RMSE': rmse, 'R2': r2}

    def plot_predictions(self, y_true, y_pred, fig_name):
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label='Actual SOC')
        plt.plot(y_pred, label='Predicted SOC')
        plt.xlabel('Sample')
        plt.ylabel('State of Charge (SOC)')
        plt.title('Actual vs Predicted SOC')
        plt.legend()
        plt.grid(True)
        # Combine directory path and figure name
        save_path = os.path.join(self.config.FIGURE_SAVE_PATH, fig_name)
        plt.savefig(save_path)
        #plt.show()
        print(f"Figure saved to {save_path}")

    def save_model(self, model_name):
        save_path = os.path.join(self.config.MODEL_SAVE_PATH, model_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {self.config.MODEL_SAVE_PATH}")


# Example usage
if __name__ == '__main__':
    # Load datasets
    train_dataset = BatterySOCDataset('../../data/soc/train/X_train.csv', '../../data/soc/train/Y_train.csv')
    test_dataset_n10 = BatterySOCDataset('../../data/soc/test/X_test_n10.csv', '../../data/soc/test/Y_test_n10.csv')
    test_dataset_0 = BatterySOCDataset('../../data/soc/test/X_test_0.csv', '../../data/soc/test/Y_test_0.csv')
    test_dataset_10 = BatterySOCDataset('../../data/soc/test/X_test_10.csv', '../../data/soc/test/Y_test_10.csv')
    test_dataset_25 = BatterySOCDataset('../../data/soc/test/X_test_25.csv', '../../data/soc/test/Y_test_25.csv')

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Initialize model and detector
    lstm_model = LSTMModel(input_size=Config.INPUT_SIZE, hidden_size=Config.HIDDEN_SIZE)
    # Initialize the TTLMLP model
    # ttl = TTLTransformerBlock(d_model=Config.D_MODEL, input_dim=Config.INPUT_DIM, output_dim=Config.OUTPUT_DIM, seq=500)
    detector = BatterySOCDetector(model=lstm_model)

    # Train the model
    detector.train(train_loader)

    # ----------------------------------------------
    # Prepare test data for n10
    X_test = test_dataset_n10.X
    y_test = test_dataset_n10.y

    # Evaluate the model
    evaluation = detector.evaluate(X_test, y_test)
    predictions = detector.predict(X_test)

    print("Predicted SOC:", predictions)
    print("Evaluation Metrics:", evaluation)

    # Plot predictions
    detector.plot_predictions(y_test.flatten(), predictions, 'soc_predictions_n10.png')

    # ----------------------------------------------
    # Prepare test data for 0
    X_test = test_dataset_0.X
    y_test = test_dataset_0.y

    # Evaluate the model
    evaluation = detector.evaluate(X_test, y_test)
    predictions = detector.predict(X_test)

    print("Predicted SOC:", predictions)
    print("Evaluation Metrics:", evaluation)

    # Plot predictions
    detector.plot_predictions(y_test.flatten(), predictions, 'soc_predictions_0.png')

    # ----------------------------------------------
    # Prepare test data for 10
    X_test = test_dataset_10.X
    y_test = test_dataset_10.y

    # Evaluate the model
    evaluation = detector.evaluate(X_test, y_test)
    predictions = detector.predict(X_test)

    print("Predicted SOC:", predictions)
    print("Evaluation Metrics:", evaluation)

    # Plot predictions
    detector.plot_predictions(y_test.flatten(), predictions, 'soc_predictions_10.png')

    # ----------------------------------------------
    # Prepare test data for 25
    X_test = test_dataset_25.X
    y_test = test_dataset_25.y

    # Evaluate the model
    evaluation = detector.evaluate(X_test, y_test)
    predictions = detector.predict(X_test)

    print("Predicted SOC:", predictions)
    print("Evaluation Metrics:", evaluation)

    # Plot predictions
    detector.plot_predictions(y_test.flatten(), predictions, 'soc_predictions_25.png')

    # Save the trained model
    detector.save_model('battery_soc_model.pth')
