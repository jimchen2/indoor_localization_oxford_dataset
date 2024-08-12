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

```sh
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
###### With higher dropout #####
# Longer sequence length
python lstm_train.py --sequence_length 200 --hidden_sizes 64 32 --num_epochs 50 --dropout_rate 0.5

# Single layer LSTM
python lstm_train.py --sequence_length 200 --hidden_sizes 64 --num_epochs 50 --dropout_rate 0.5
```

## Tensorboard Results

![Image](https://blog.jimchen.me/2d5064e9-4cd2-4297-8216-36ed52e18c5c-1723134722726.jpg)
![Image](https://blog.jimchen.me/c6142212-c6c6-4fc2-92d8-94983953e1a4-1723134741631.jpg)
![Image](https://blog.jimchen.me/05224979-bb11-41ae-bcdc-d04cc9d25335-1723134773118.jpg)
![Image](https://blog.jimchen.me/186cd032-eeae-4ad1-81c9-ef285b926dba-1723134801099.jpg)

## Some More Training and Finetuning


```sh
python lstm_train.py --sequence_length 200 --hidden_sizes 32 --num_epochs 80 --dropout_rate 0.5
python lstm_train.py --sequence_length 200 --hidden_sizes 32 --num_epochs 50 --dropout 0.2
# Large Batch
python lstm_train.py --sequence_length 200 --hidden_sizes 32 --num_epochs 100 --batch_size 128 --dropout_rate 0.2
python lstm_train.py --sequence_length 200 --hidden_sizes 32 --num_epochs 100 --batch_size 128 --dropout_rate 0.4
# Large Hidden Size
python lstm_train.py --sequence_length 200 --hidden_sizes 64 --num_epochs 50 --dropout_rate 0.2

# Run it twice
python lstm_train.py --sequence_length 200 --hidden_sizes 64 --num_epochs 80 --dropout_rate 0.4
python lstm_train.py --sequence_length 200 --hidden_sizes 64 --num_epochs 80 --dropout_rate 0.4

python lstm_train.py --sequence_length 200 --hidden_sizes 256 --num_epochs 50 --dropout_rate 0.4

# More Runs
python lstm_train.py --sequence_length 200 --hidden_sizes 64 --num_epochs 60 --dropout_rate 0.3
python lstm_train.py --sequence_length 200 --hidden_sizes 96 --num_epochs 80 --dropout_rate 0.4
python lstm_train.py --sequence_length 200 --hidden_sizes 128 64 --num_epochs 75 --dropout_rate 0.4
```

## Tensorboard Results

![Image](https://blog.jimchen.me/7ff27652-ad82-4d95-829c-6c928fa42811-1723135025275.jpg)
![Image](https://blog.jimchen.me/d719be18-fdc9-459e-8b22-f9ad17f082b0-1723135041582.jpg)
![Image](https://blog.jimchen.me/81780849-a262-4108-a9da-8e4dd6f3e8f0-1723135083765.jpg)
![Image](https://blog.jimchen.me/567b89f7-5b8a-4acb-b2ea-d1c9e06e6ae6-1723135062276.jpg) 

## Extreme Long Seq Length

I also tried to increase the sequence length to 400, which resulted in 
```
Overall Mean Squared Error: 0.2241
Overall Mean Absolute Error: 0.2934
```

So like I cannot get it much below 0.3 with this vanilla LSTM approach