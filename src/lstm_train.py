import argparse
import torch
from lstm_model import IMULSTMModel
from data_preprocessing import prepare_data

def train(args):
    # Load data
    train_loader, val_loader, train_dataset, val_dataset = prepare_data(args.root_dir, args.sequence_length, args.batch_size)

    # Print data information
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    print(f"Total samples: {len(train_dataset) + len(val_dataset)}")
    print(f"Input shape: {train_dataset[0][0].shape}")
    print(f"Target shape: {train_dataset[0][1].shape}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Initialize the model, loss function, and optimizer
    model = IMULSTMModel(args.input_size, args.hidden_size, args.num_layers, args.output_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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
    for epoch in range(args.num_epochs):
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
        
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_save_path)

    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model for IMU data")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--sequence_length", type=int, default=100, help="Sequence length for LSTM input")
    parser.add_argument("--input_size", type=int, default=15, help="Number of features in IMU data")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--output_size", type=int, default=3, help="Output size (x, y, z)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--model_save_path", type=str, default="../model/best_imu_lstm_model.pth", help="Path to save the best model")

    args = parser.parse_args()
    train(args)