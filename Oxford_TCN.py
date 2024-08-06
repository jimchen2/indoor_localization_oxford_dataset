import h5py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import random
import numpy as np
import argparse
import json
from datetime import datetime

# Configuration
CONFIG = {
    'window_size': 120,
    'batch_size': 32,
    'num_epochs': 1000,
    'patience': 10,
    'learning_rate': 0.001,
    'seed': 42,
    'root_directory': 'Oxford Inertial Odometry Dataset',
    'data_folder': 'data_hdf5',
    'num_runs': 5  # Number of times to train the model
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class MagneticDataset(Dataset):
    def __init__(self, mag_data, positions):
        self.mag_data = torch.tensor(mag_data, dtype=torch.float32)
        self.positions = torch.tensor(positions, dtype=torch.float32)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.mag_data[idx], self.positions[idx]

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same', dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        return self.leaky_relu(out + residual)

class MagneticModel(nn.Module):
    def __init__(self, input_shape, nb_filters=128):
        super(MagneticModel, self).__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_shape[1], nb_filters, kernel_size=3, dilation=1),
            TCNBlock(nb_filters, nb_filters, kernel_size=3, dilation=2),
            TCNBlock(nb_filters, nb_filters, kernel_size=3, dilation=4),
            TCNBlock(nb_filters, nb_filters, kernel_size=3, dilation=8),
            TCNBlock(nb_filters, nb_filters, kernel_size=3, dilation=16),
            TCNBlock(nb_filters, nb_filters, kernel_size=3, dilation=32)
        )

        self.flatten = nn.Flatten()

        self.fc_layers = nn.Sequential(
            nn.Linear(nb_filters * input_shape[0], 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.1),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, channels, sequence_length)
        x = self.tcn(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

def load_data(config, dataset_name):
    file_path = os.path.join(config['data_folder'], f'{dataset_name}.hdf5')
    
    with h5py.File(file_path, 'r') as f:
        train_data = torch.from_numpy(f['train_data'][:])
        train_labels = torch.from_numpy(f['train_labels'][:])
        val_data = torch.from_numpy(f['val_data'][:])
        val_labels = torch.from_numpy(f['val_labels'][:])
        test_data = torch.from_numpy(f['test_data'][:])
        test_labels = torch.from_numpy(f['test_labels'][:])
    
    print("Training data shape:", train_data.shape, train_labels.shape)
    print("Validation data shape:", val_data.shape, val_labels.shape)
    print("Testing data shape:", test_data.shape, test_labels.shape)
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def create_data_loaders(train_data, train_labels, val_data, val_labels, test_data, test_labels, config):
    train_dataset = MagneticDataset(train_data, train_labels)
    val_dataset = MagneticDataset(val_data, val_labels)
    test_dataset = MagneticDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader


def train(model, criterion, optimizer, scheduler, device, train_loader, val_loader, config, run_dir):
    best_val_loss = float('inf')
    pbar = tqdm(range(config['num_epochs']))
    no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in pbar:
        # Train
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_mae_loss = train_loss / len(train_loader)
        train_losses.append(train_mae_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_mag, batch_pos in val_loader:
                batch_mag, batch_pos = batch_mag.to(device), batch_pos.to(device)
                outputs = model(batch_mag)
                loss = criterion(outputs, batch_pos)
                val_loss += loss.item()

        val_mae_loss = val_loss / len(val_loader)
        val_losses.append(val_mae_loss)

        pbar.set_postfix({
            'Epoch': epoch+1,
            'Train MAE Loss': f'{train_mae_loss:.4f}',
            'Val MAE Loss': f'{val_mae_loss:.4f}'
        })

        scheduler.step(val_mae_loss)

        # Save the best model
        if val_mae_loss < best_val_loss:
            best_val_loss = val_mae_loss
            model_save_path = os.path.join(run_dir, f'model_mae{val_mae_loss:.4f}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_mae_loss,
                'val_loss': val_mae_loss,
            }, model_save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve == config['patience']:
                print("\nEarly stopping!")
                break

    return train_losses, val_losses, best_val_loss, model_save_path



def evaluate(model, criterion, device, test_loader):
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_mae_loss = test_loss / len(test_loader)
    print(f'Test MAE: {test_mae_loss:.4f}')
    return test_mae_loss, np.array(all_predictions), np.array(all_targets)


def create_run_directory(main_dir, run_id, dataset_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(main_dir, f'run_{dataset_name}_{run_id}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main(args):
    set_seed(CONFIG['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data(CONFIG, args.dataset)
    train_loader, val_loader, test_loader = create_data_loaders(train_data, train_labels, val_data, val_labels, test_data, test_labels, CONFIG)

    # Create a main directory for all runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_dir = f'experiment_{args.dataset}_{CONFIG["window_size"]}_{timestamp}'
    os.makedirs(main_dir, exist_ok=True)
    
    # Create a log file
    log_file = os.path.join(main_dir, f'log_{args.dataset}_{timestamp}.txt')
    
    all_runs_info = []

    for run in range(CONFIG['num_runs']):
        print(f"\nStarting Run {run + 1}/{CONFIG['num_runs']}")
        
        # Create a directory for this run
        run_dir = create_run_directory(main_dir, run+1, args.dataset)
        
        
        model = MagneticModel(input_shape=(CONFIG['window_size'], 3))
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        criterion = nn.L1Loss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=CONFIG['patience'], factor=0.1)

        train_losses, val_losses, best_val_loss, model_save_path = train(model, criterion, optimizer, scheduler, device, train_loader, val_loader, CONFIG, run_dir)

        # Load best model for evaluation
        model.load_state_dict(torch.load(model_save_path)['model_state_dict'])
        test_mae_loss, predictions, targets = evaluate(model, criterion, device, test_loader)
        
        run_info = {
            'run': run + 1,
            'best_val_loss': best_val_loss,
            'test_mae_loss': test_mae_loss,
            'model_path': model_save_path,
            'run_directory': run_dir
        }
        all_runs_info.append(run_info)
        
        # Save predictions and targets
        np.save(os.path.join(run_dir, 'predictions.npy'), predictions)
        np.save(os.path.join(run_dir, 'targets.npy'), targets)
        
        # Save losses
        np.save(os.path.join(run_dir, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(run_dir, 'val_losses.npy'), np.array(val_losses))

    # Write all runs info to the log file
    with open(log_file, 'w') as f:
        json.dump(all_runs_info, f, indent=4)

    print(f"All runs completed. Results saved in {main_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate the magnetic model')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset file (without extension)')
    args = parser.parse_args()
    main(args)