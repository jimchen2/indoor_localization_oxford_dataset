import argparse
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL ERRS ONLY
from transformer_model import IMUTransformerModel
from data_preprocessing import prepare_data
from transformer_test import test_model
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

def train_and_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("../../transformer_logs", f"{current_time}_d_model{args.d_model}_nhead{args.nhead}_layers{args.num_layers}_ff{args.dim_feedforward}_output{args.output_size}_batch{args.batch_size}_dropout{args.dropout_rate}_sequencelength{args.sequence_length}_pooling{args.pooling}_returnall{args.return_all_positions}")    
    writer = SummaryWriter(log_dir)

    # Load data
    train_loader, val_loader, train_dataset, val_dataset = prepare_data(args.root_dir, args.sequence_length, args.batch_size, device)

    # Print and log data information
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    print(f"Total samples: {len(train_dataset) + len(val_dataset)}")
    print(f"Input shape: {train_dataset[0][0].shape}")
    print(f"Target shape: {train_dataset[0][1].shape}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    writer.add_text("Dataset Info", f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    writer.add_text("Input Shape", str(train_dataset[0][0].shape))
    writer.add_text("Target Shape", str(train_dataset[0][1].shape))

    # Initialize the model, loss function, and optimizer
    model = IMUTransformerModel(
        input_size=args.input_size, 
        d_model=args.d_model, 
        nhead=args.nhead, 
        num_layers=args.num_layers, 
        dim_feedforward=args.dim_feedforward, 
        output_size=args.output_size, 
        dropout=args.dropout_rate
    ).to(device)
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

    # Log baseline loss as "Epoch 0"
    writer.add_scalar("Loss/Train", baseline_train_loss, 0)
    writer.add_scalar("Loss/Validation", baseline_val_loss, 0)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (imu_seq, vi_target) in enumerate(train_loader):
            imu_seq, vi_target = imu_seq.to(device), vi_target.to(device)
            outputs = model(imu_seq, return_all_positions=args.return_all_positions, pooling=args.pooling)
            if args.return_all_positions:
                loss = criterion(outputs, vi_target.unsqueeze(1).expand(-1, outputs.size(1), -1))
            else:
                loss = criterion(outputs, vi_target)
                
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imu_seq, vi_target in val_loader:
                imu_seq, vi_target = imu_seq.to(device), vi_target.to(device)
                outputs = model(imu_seq, return_all_positions=args.return_all_positions, pooling=args.pooling)
                if args.return_all_positions:
                    val_loss += criterion(outputs, vi_target.unsqueeze(1).expand(-1, outputs.size(1), -1)).item()
                else:
                    val_loss += criterion(outputs, vi_target).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Log to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_save_path)
            writer.add_scalar("Best_Val_Loss", best_val_loss, epoch + 1)

    print("Training completed.")

    # Testing
    print("\nStarting model evaluation...")
    
    # Load the best model
    model.load_state_dict(torch.load(args.model_save_path, map_location=device, weights_only=True))
    model.eval()

    test_loss, overall_mse, overall_mae = test_model(model, args.test_root_dir, args.sequence_length, args.output_size, device)

    # Log test results to TensorBoard
    writer.add_scalar("Test/Loss", test_loss, 0)
    writer.add_scalar("Test/MSE", overall_mse, 0)
    writer.add_scalar("Test/MAE", overall_mae, 0)

    # Add text summary of test results
    writer.add_text("Test Results", 
                    f"Test Loss: {test_loss:.4f}\n"
                    f"Overall MSE: {overall_mse:.4f}\n"
                    f"Overall MAE: {overall_mae:.4f}")

    print(f"Test results logged to TensorBoard in {log_dir}")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Transformer model for IMU data")
    parser.add_argument("--root_dir", type=str, default="../../data/Oxford Inertial Odometry Dataset/handheld", help="Root directory of the dataset")
    parser.add_argument("--sequence_length", type=int, default=200, help="Sequence length for Transformer input")
    parser.add_argument("--input_size", type=int, default=12, help="Number of features in IMU data")
    parser.add_argument("--d_model", type=int, default=64, help="Dimension of the model")
    parser.add_argument("--nhead", type=int, default=2, help="Number of heads in multi-head attention")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="Dimension of feedforward network")
    parser.add_argument("--output_size", type=int, default=3, help="Output size (x, y, z)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--model_save_path", type=str, default="../../model/best_imu_transformer_model.pth", help="Path to save the best model")
    parser.add_argument("--test_root_dir", type=str, default="../../data/Oxford Inertial Odometry Dataset/handheld_test", help="Root directory of the test dataset")
    parser.add_argument("--pooling", type=str, choices=['last', 'mean'], default='last', help="Pooling method for output (last or mean)")
    parser.add_argument("--return_all_positions", action="store_true", help="Return predictions for all positions in the sequence")

    args = parser.parse_args()
    train_and_evaluate(args)