## Resnet for Localization

I removed the attitude data



## Commands to run


Dummy Script

```bash
python resnet_train.py --num_epochs 50
```


```bash

# Baseline configuration
python resnet_train.py --sequence_length 200 --input_size 12 --channels 64 128 256 --output_size 3 --learning_rate 0.001 --batch_size 32 --num_epochs 100 --dropout_rate 0.3

# Longer sequence length
python resnet_train.py --sequence_length 300 --input_size 12 --channels 64 128 256 --output_size 3 --learning_rate 0.001 --batch_size 32 --num_epochs 100 --dropout_rate 0.3

# Smaller learning rate
python resnet_train.py --sequence_length 200 --input_size 12 --channels 64 128 256 --output_size 3 --learning_rate 0.0005 --batch_size 32 --num_epochs 150 --dropout_rate 0.3

# Larger batch size
python resnet_train.py --sequence_length 200 --input_size 12 --channels 64 128 256 --output_size 3 --learning_rate 0.001 --batch_size 64 --num_epochs 100 --dropout_rate 0.3

# Higher dropout rate
python resnet_train.py --sequence_length 200 --input_size 12 --channels 64 128 256 --output_size 3 --learning_rate 0.001 --batch_size 32 --num_epochs 100 --dropout_rate 0.5

# Deeper network
python resnet_train.py --sequence_length 200 --input_size 12 --channels 64 128 256 512 --output_size 3 --learning_rate 0.001 --batch_size 32 --num_epochs 120 --dropout_rate 0.3

# Wider network
python resnet_train.py --sequence_length 200 --input_size 12 --channels 128 256 512 --output_size 3 --learning_rate 0.001 --batch_size 32 --num_epochs 100 --dropout_rate 0.3

# Shorter sequence length with larger batch size
python resnet_train.py --sequence_length 150 --input_size 12 --channels 64 128 256 --output_size 3 --learning_rate 0.001 --batch_size 128 --num_epochs 100 --dropout_rate 0.3

# Longer sequence length with smaller learning rate
python resnet_train.py --sequence_length 400 --input_size 12 --channels 64 128 256 --output_size 3 --learning_rate 0.0002 --batch_size 32 --num_epochs 200 --dropout_rate 0.3

# Shallow network with higher dropout
python resnet_train.py --sequence_length 200 --input_size 12 --channels 128 256 --output_size 3 --learning_rate 0.001 --batch_size 32 --num_epochs 150 --dropout_rate 0.6
```


# ResNet-like 1D CNN Architecture

## Input Block

- BatchNorm1d (input_size channels)
- 1D Conv (input_size -> channels[0], kernel_size 7, stride 2)
- BatchNorm1d
- ReLU
- 1D MaxPool (kernel_size 3, stride 2)

## Residual Blocks

Multiple layers of ResidualBlocks, with increasing channel sizes:

- Each layer contains 2 ResidualBlocks
- ResidualBlock structure:
  1. 1D Conv (in_channels -> out_channels, kernel_size 3, stride 1 or 2)
  2. BatchNorm1d
  3. ReLU
  4. 1D Conv (out_channels -> out_channels, kernel_size 3, stride 1)
  5. BatchNorm1d
  6. Skip connection (with 1x1 Conv if needed)
  7. ReLU

## Output Block

- Adaptive Average Pooling (to 1)
- Flatten
- Dropout
- Fully Connected layer (channels[-1] -> output_size)