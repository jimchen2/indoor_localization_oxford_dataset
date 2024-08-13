import argparse
import torch
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
from resnet_model import IMUResNetModel
from data_preprocessing import prepare_data
from resnet_test import test_model
from torch.utils.tensorboard import SummaryWriter
import datetime

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate(args):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("../../other_exp_logs", f"{current_time}_input{args.input_size}_channels{'-'.join(map(str, args.channels))}_output{args.output_size}_lr{args.learning_rate}_batch{args.batch_size}_dropout{args.dropout_rate}_sequencelength{args.sequence_length}")    
    writer = SummaryWriter(log_dir)

    train_loader, val_loader, train_dataset, val_dataset = prepare_data(args.root_dir, args.sequence_length, args.batch_size)

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

    device = get_device()
    print(f"Using device: {device}")

    # Define dropout rates for different parts of the network
    dropout_rates = {
        'input': args.dropout_rate,
        'block1': [args.dropout_rate] * len(args.group_sizes),
        'block2': [args.dropout_rate] * len(args.group_sizes),
        'output': args.dropout_rate
    }

    model = IMUResNetModel(args.input_size, args.channels, args.output_size, dropout_rates, args.group_sizes).to(device)    
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

    all_targets = torch.cat([targets for _, targets in train_loader], dim=0)
    mean_position = all_targets.mean(dim=0)

    print("Performing mean baseline evaluation...")
    baseline_train_loss = mean_baseline_eval(train_loader, mean_position)
    baseline_val_loss = mean_baseline_eval(val_loader, mean_position)
    print(f"Baseline Train Loss: {baseline_train_loss:.4f}, Baseline Val Loss: {baseline_val_loss:.4f}")

    writer.add_scalar("Loss/Train", baseline_train_loss, 0)
    writer.add_scalar("Loss/Validation", baseline_val_loss, 0)

    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (imu_seq, vi_target) in enumerate(train_loader):
            imu_seq, vi_target = imu_seq.to(device), vi_target.to(device)
            
            # Transpose the input tensor
            imu_seq = imu_seq.transpose(1, 2)
            
            outputs = model(imu_seq)
            loss = criterion(outputs, vi_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imu_seq, vi_target in val_loader:
                imu_seq, vi_target = imu_seq.to(device), vi_target.to(device)
                
                # Transpose the input tensor
                imu_seq = imu_seq.transpose(1, 2)
                
                outputs = model(imu_seq)
                val_loss += criterion(outputs, vi_target).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.cpu().state_dict(), args.model_save_path)
            model.to(device)
            writer.add_scalar("Best_Val_Loss", best_val_loss, epoch + 1)

    print("Training completed.")

    print("\nStarting model evaluation...")
    
    model.load_state_dict(torch.load(args.model_save_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    test_loss, predictions, targets, overall_metrics = test_model(model, args.test_root_dir, args.sequence_length, args.output_size)

    for metric, value in overall_metrics.items():
        writer.add_scalar(f"Test/{metric}", value, 0)

    writer.add_histogram("Test/Predictions", predictions, 0)
    writer.add_histogram("Test/Targets", targets, 0)

    print(f"Test results logged to TensorBoard in {log_dir}")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ResNet model for IMU data")
    parser.add_argument("--root_dir", type=str, default="../../data/Oxford Inertial Odometry Dataset/handheld", help="Root directory of the dataset")
    parser.add_argument("--sequence_length", type=int, default=200, help="Sequence length for ResNet input")
    parser.add_argument("--input_size", type=int, default=12, help="Number of features in IMU data")
    parser.add_argument("--channels", type=int, nargs='+', default=[64, 128, 256], help="Number of channels in ResNet layers")
    parser.add_argument("--output_size", type=int, default=3, help="Output size (x, y, z)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--dropout_rate", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--model_save_path", type=str, default="../../model/best_imu_resnet_model.pth", help="Path to save the best model")
    parser.add_argument("--test_root_dir", type=str, default="../../data/Oxford Inertial Odometry Dataset/handheld_test", help="Root directory of the test dataset")
    parser.add_argument("--group_sizes", type=int, nargs='+', default=[2, 2, 2], help="Group sizes for ResNet layers")

    args = parser.parse_args()
    train_and_evaluate(args)