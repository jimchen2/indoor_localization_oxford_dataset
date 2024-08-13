## Resnet for Localization

I removed the attitude data



## Commands to run


```bash
python resnet_train.py --num_epochs 50

```

## Hyperparams in ResNet

Input Block:

    7x7 Conv, 64 filters, stride 2
    BatchNorm
    ReLU
    3x3 Max Pooling, stride 2
    Dropout 

Many Blocks (each block is either Bottleneck or Basic, with increasing channels and decreasing spatial dimensions across stages)

Bottleneck:

    1x1 Conv (input channels -> reduced channels)
    BatchNorm -> ReLU
    3x3 Conv (reduced channels -> reduced channels)
    BatchNorm -> ReLU
    1x1 Conv (reduced channels -> increased channels)
    BatchNorm
    Add input to the output (skip connection, with projection if needed)
    ReLU
    Dropout

Basic:

    3x3 Conv (input channels -> output channels)
    BatchNorm -> ReLU
    Dropout 
    3x3 Conv (output channels -> output channels)
    BatchNorm
    Add input to the output (skip connection, with projection if needed)
    ReLU
    Dropout 

Output Block:

    Global Average Pooling
    Dropout
    Fully Connected layer (num_features -> num_classes)
    Softmax activation



```
hyperparams = {
    # Architecture
    'num_blocks': [3, 4, 6, 3],  # Number of blocks in each stage
    'block_type': ['Bottleneck', 'Basic'],  # Choose block type for each stage
    'channels': [64, 128, 256, 512],  # Channels in each stage

    # Input Block
    'initial_filters': 64,
    'initial_kernel_size': 7,
    'initial_stride': 2,
    'pool_size': 3,
    'pool_stride': 2,

    # Dropout
    'dropout_rate_conv': 0.1,
    'dropout_rate_fc': 0.5,

    # Training
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'optimizer': 'Adam',
    'weight_decay': 0.0001,

    # Data Augmentation
    'use_augmentation': True,
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
}
```