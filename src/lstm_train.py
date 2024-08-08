import torch
from lstm_model import IMULSTMModel
from data_preprocessing import prepare_data

# Hyperparameters
sequence_length = 100
input_size = 15  # Number of features in IMU data
hidden_size = 64
num_layers = 2
output_size = 3  # x, y, z
learning_rate = 0.001
batch_size = 32
num_epochs = 50

# Load data
root_dir = '/home/user/Code/magnetic_localization/data/Oxford Inertial Odometry Dataset/trolley'
train_loader, val_loader, train_dataset, val_dataset = prepare_data(root_dir, sequence_length, batch_size)

# Print data information
print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(val_dataset)}")
print(f"Total samples: {len(train_dataset) + len(val_dataset)}")
print(f"Input shape: {train_dataset[0][0].shape}")
print(f"Target shape: {train_dataset[0][1].shape}")
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