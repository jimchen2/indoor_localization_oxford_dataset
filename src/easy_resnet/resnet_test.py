import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from metrics import calculate_metrics

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IMUSequence:
    def __init__(self, imu_data, vi_data, name):
        self.imu_data = imu_data
        self.vi_data = vi_data
        self.name = name

def load_data(root_dir):
    sequences = []
    for data_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, data_folder, 'syn')
        if os.path.isdir(folder_path):
            imu_files = sorted([f for f in os.listdir(folder_path) if f.startswith('imu')])
            vi_files = sorted([f for f in os.listdir(folder_path) if f.startswith('vi')])
            
            for imu_file, vi_file in zip(imu_files, vi_files):
                imu = pd.read_csv(os.path.join(folder_path, imu_file), header=None).iloc[:, list(range(4,16))].values
                vi = pd.read_csv(os.path.join(folder_path, vi_file), header=None).iloc[:, 2:5].values
                sequences.append(IMUSequence(imu, vi, f"{data_folder}/{imu_file}"))
    return sequences


def evaluate_model(model, sequences, sequence_length, output_size):
    device = get_device()
    model.to(device)
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []    
    with torch.no_grad():
        for seq_idx, sequence in enumerate(tqdm(sequences, desc="Evaluating")):
            print("Start evaluating")
            seq_predictions = []
            seq_targets = sequence.vi_data
            
            for i in range(len(sequence.imu_data)):
                if i < sequence_length - 1:
                    pad_length = sequence_length - 1 - i
                    imu_seq = np.pad(sequence.imu_data[:i+1], ((pad_length, 0), (0, 0)), mode='constant')
                else:
                    imu_seq = sequence.imu_data[i-sequence_length+1:i+1]
                
                # Prepare the input tensor
                imu_seq = torch.FloatTensor(imu_seq).unsqueeze(0).to(device)
                imu_seq = imu_seq.transpose(1, 2)  # Transpose to [1, 12, 200]
                
                # Ensure the input has the correct shape
                if imu_seq.shape != (1, 12, sequence_length):
                    imu_seq = imu_seq.squeeze()  # Remove any extra dimensions
                    if imu_seq.shape != (12, sequence_length):
                        raise ValueError(f"Unexpected input shape: {imu_seq.shape}")
                    imu_seq = imu_seq.unsqueeze(0)  # Add batch dimension back
                
                output = model(imu_seq).squeeze(0)
                
                seq_predictions.append(output.cpu().numpy())
                
                loss = torch.nn.functional.mse_loss(output, torch.FloatTensor(seq_targets[i]).to(device))
                total_loss += loss.item()
            
            seq_predictions = np.array(seq_predictions)
            seq_targets = np.array(seq_targets)
            
            seq_metrics = calculate_metrics(seq_predictions, seq_targets)
            
            print(f"\nSequence: {sequence.name}")
            for metric, value in seq_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            all_predictions.extend(seq_predictions)
            all_targets.extend(seq_targets)
    
    avg_loss = total_loss / sum(len(seq.imu_data) for seq in sequences)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    overall_metrics = calculate_metrics(all_predictions, all_targets)
        
    return avg_loss, all_predictions, all_targets, overall_metrics


def test_model(model, test_root_dir, sequence_length, output_size):
    print("\nStarting model evaluation...")

    sequences = load_data(test_root_dir)

    print("Testing Dataset Information:")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Model Sequence length: {sequence_length}")

    test_loss, predictions, targets, overall_metrics = evaluate_model(model, sequences, sequence_length, output_size)
    print(f"\nOverall Test Loss: {test_loss:.4f}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    print("\nOverall Metrics:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")

    return test_loss, predictions, targets, overall_metrics