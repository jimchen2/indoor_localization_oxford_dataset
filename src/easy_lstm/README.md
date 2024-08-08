## Easy LSTM

So basically this is an intuitive file for LSTM, with minimal configurations and it can be trained 5 minutes on a CPU

We basically did a really simple thing, like feed everything into LSTM model (split data on the file level)

## Handheld Training

As we can see it quickly reaches some benchmark(though not bad)

```
Total training samples: 6002
Total validation samples: 981
Total samples: 6983
Input shape: torch.Size([100, 15])
Target shape: torch.Size([3])
Number of training batches: 188
Number of validation batches: 31
Performing mean baseline evaluation...
Baseline Train Loss: 1.5063, Baseline Val Loss: 1.6076
Epoch [10/50], Train Loss: 0.5113, Val Loss: 0.5460
Epoch [15/50], Train Loss: 0.3930, Val Loss: 0.4855
Epoch [20/50], Train Loss: 0.3466, Val Loss: 0.3950
Epoch [25/50], Train Loss: 0.3294, Val Loss: 0.4008
Epoch [27/50], Train Loss: 0.3010, Val Loss: 0.3548
```

## Tried it on Trolley

```
(mlenv) user@fedora ~/C/m/src (master)> python easy_lstm.py
Total training samples: 3880
Total validation samples: 376
Total samples: 4256
Input shape: torch.Size([100, 15])
Target shape: torch.Size([3])
Number of training batches: 122
Number of validation batches: 12
Performing mean baseline evaluation...
Baseline Train Loss: 1.6448, Baseline Val Loss: 1.4999
Epoch [5/50], Train Loss: 0.5197, Val Loss: 0.4581
Epoch [10/50], Train Loss: 0.5856, Val Loss: 0.4645
Epoch [15/50], Train Loss: 0.4282, Val Loss: 0.4330
Epoch [20/50], Train Loss: 0.4565, Val Loss: 0.4461
Epoch [25/50], Train Loss: 0.3967, Val Loss: 0.3775
Epoch [29/50], Train Loss: 0.3408, Val Loss: 0.3440
```

## Finetuning

We need to finetune these

Basically I run these to test 

```python
# Basic configuration:
python lstm_train.py --sequence_length 100 --hidden_sizes 64 32 --num_epochs 50
# Longer sequence length
python lstm_train.py --sequence_length 200 --hidden_sizes 64 32 --num_epochs 50
# Shorter sequence length
python lstm_train.py --sequence_length 50 --hidden_sizes 64 32 --num_epochs 50
# Single layer LSTM
python lstm_train.py --sequence_length 100 --hidden_sizes 128 --num_epochs 50
# Three-layer LSTM
python lstm_train.py --sequence_length 100 --hidden_sizes 64 32 16 --num_epochs 50
# Larger hidden sizes
python lstm_train.py --sequence_length 100 --hidden_sizes 128 64 --num_epochs 50
# Smaller hidden sizes
python lstm_train.py --sequence_length 100 --hidden_sizes 32 16 --num_epochs 50
# Complex configuration
python lstm_train.py --sequence_length 300 --hidden_sizes 128 64 32 16 --num_epochs 75
```

More Runs with higher dropout

```python
# Longer sequence length
python lstm_train.py --sequence_length 200 --hidden_sizes 64 32 --num_epochs 50 --dropout_rate 0.5

# Single layer LSTM
python lstm_train.py --sequence_length 200 --hidden_sizes 64 --num_epochs 50 --dropout_rate 0.5

# Single layer 256 hidden size LSTM
python lstm_train.py --sequence_length 200 --hidden_sizes 256 --num_epochs 50 --dropout_rate 0.5

# Three-layer LSTM
python lstm_train.py --sequence_length 200 --hidden_sizes 128 64 32 --num_epochs 50 --dropout_rate 0.5
```

## Reports

![Image](https://blog.jimchen.me/8e6da0ef-e749-440d-8a24-939d726c81e2-1723115444353.jpg)
![Image](https://blog.jimchen.me/7abb9b40-a8d4-4d02-be89-81e4deaecef1-1723115455459.jpg)
![Image](https://blog.jimchen.me/c5758ecf-8507-4470-9491-cde9d9e70611-1723115640143.jpg)
![Image](https://blog.jimchen.me/f88b4c2b-d995-4765-86a4-2e0a36a43d83-1723115648368.jpg)

