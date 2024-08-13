## Resnet for Localization

I removed the attitude data



## Commands to run


```bash
# 1. Basic run
python resnet_train.py --num_epochs 100 --channels 16 32 64 --group_sizes 2 2 2 --dropout_rate 0.2
# 2. Increase epochs
python resnet_train.py --num_epochs 200 --channels 16 32 64 --group_sizes 2 2 2 --dropout_rate 0.2
# 3. Decrease epochs
python resnet_train.py --num_epochs 50 --channels 16 32 64 --group_sizes 2 2 2 --dropout_rate 0.2
# 4. Increase channels
python resnet_train.py --num_epochs 100 --channels 32 64 128 --group_sizes 2 2 2 --dropout_rate 0.2
# 5. Decrease channels
python resnet_train.py --num_epochs 100 --channels 8 16 32 --group_sizes 2 2 2 --dropout_rate 0.2
# 6. Increase group sizes
python resnet_train.py --num_epochs 100 --channels 16 32 64 --group_sizes 3 3 3 --dropout_rate 0.2
# 7. Decrease group sizes
python resnet_train.py --num_epochs 100 --channels 16 32 64 --group_sizes 1 1 1 --dropout_rate 0.2
# 8. Increase dropout rate
python resnet_train.py --num_epochs 100 --channels 16 32 64 --group_sizes 2 2 2 --dropout_rate 0.4
# 9. Decrease dropout rate
python resnet_train.py --num_epochs 100 --channels 16 32 64 --group_sizes 2 2 2 --dropout_rate 0.1
# 10. Mix of changes
python resnet_train.py --num_epochs 150 --channels 24 48 96 --group_sizes 2 3 2 --dropout_rate 0.3
```

## Hyperparams in 1D ResNet

Input Block:

    1D Conv, base_plane filters, kernel_size 7, stride 2
    BatchNorm1D
    ReLU
    1D Max Pooling, kernel_size 3, stride 2

Many Blocks (BasicBlock1D, with increasing channels across stages)

BasicBlock1D:

    3x1 Conv1D (in_planes -> out_planes)
    BatchNorm1D -> ReLU
    3x1 Conv1D (out_planes -> out_planes)
    BatchNorm1D
    Add input to the output (skip connection, with projection if needed)
    ReLU

Output Block:
    Adaptive Average Pooling
    Flatten
    Fully Connected layer (in_planes -> fc_dim)
    ReLU
    Dropout
    Fully Connected layer (fc_dim -> fc_dim)
    ReLU
    Dropout
    Fully Connected layer (fc_dim -> num_outputs)