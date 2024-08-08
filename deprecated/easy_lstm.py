import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class IMUSequence:
    def __init__(self, imu_file, vi_file, sequence_length):
        imu_data = pd.read_csv(imu_file, header=None).iloc[:, 1:].values
        vi_data = pd.read_csv(vi_file, header=None).iloc[:, 2:5].values  # Only x, y, z

        self.sequences = []
        self.targets = []

        for i in range(0, len(imu_data) - sequence_length, sequence_length):
            self.sequences.append(imu_data[i:i+sequence_length])
            self.targets.append(vi_data[i+sequence_length])

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)

class IMUDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = []
        self.targets = []
        for seq in sequences:
            self.sequences.extend(seq.sequences)
            self.targets.extend(seq.targets)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

def load_sequences(root_dir, sequence_length):
    sequences = []
    for data_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, data_folder, 'syn')
        if os.path.isdir(folder_path):
            imu_files = sorted([f for f in os.listdir(folder_path) if f.startswith('imu')])
            vi_files = sorted([f for f in os.listdir(folder_path) if f.startswith('vi')])
            
            for imu_file, vi_file in zip(imu_files, vi_files):
                sequences.append(IMUSequence(
                    os.path.join(folder_path, imu_file),
                    os.path.join(folder_path, vi_file),
                    sequence_length
                ))
    return sequences

class IMULSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(IMULSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
sequence_length = 100
input_size = 15  # Number of features in IMU data
hidden_size = 64
num_layers = 2
output_size = 3  # x, y, z
learning_rate = 0.001
batch_size = 32
num_epochs = 50

# Load all sequences
root_dir = '/home/user/Code/magnetic_localization/data/Oxford Inertial Odometry Dataset/trolley'
all_sequences = load_sequences(root_dir, sequence_length)

# Split sequences into train and validation at the file level
train_sequences, val_sequences = train_test_split(all_sequences, test_size=0.1, random_state=42)

# Create datasets
train_dataset = IMUDataset(train_sequences)
val_dataset = IMUDataset(val_sequences)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)



# Print total amount of data
print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(val_dataset)}")
print(f"Total samples: {len(train_dataset) + len(val_dataset)}")

# If you want to print the shape of the data
print(f"Input shape: {train_dataset[0][0].shape}")
print(f"Target shape: {train_dataset[0][1].shape}")

# If you want to print the number of batches
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")



# Initialize the model, loss function, and optimizer
model = IMULSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def mean_baseline_eval(loader, mean_position):
    total_loss = 0
    for _, targets in loader:
        batch_size = targets.size(0)
        mean_predictions = mean_position.repeat(batch_size, 1)
        loss = torch.nn.functional.mse_loss(mean_predictions, targets)
        total_loss += loss.item() * batch_size
    return total_loss / len(loader.dataset)

# Calculate mean position from the training set
all_targets = torch.cat([targets for _, targets in train_loader], dim=0)
mean_position = all_targets.mean(dim=0)
# Perform mean baseline evaluation
print("Performing mean baseline evaluation...")
baseline_train_loss = mean_baseline_eval(train_loader, mean_position)
baseline_val_loss = mean_baseline_eval(val_loader, mean_position)
print(f"Baseline Train Loss: {baseline_train_loss:.4f}, Baseline Val Loss: {baseline_val_loss:.4f}")

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (imu_seq, vi_target) in enumerate(train_loader):
        outputs = model(imu_seq)
        loss = criterion(outputs, vi_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imu_seq, vi_target in val_loader:
            outputs = model(imu_seq)
            val_loss += criterion(outputs, vi_target).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '../model/best_imu_lstm_model.pth')

print("Training completed.")