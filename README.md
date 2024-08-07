## Setup

`wget https://public.jimchen.me/other/Oxford%20Inertial%20Odometry%20Dataset_2.0.zip`

In each data fold, there is a raw data subfolder and a syn data subfolder, which represent the raw data collection without synchronisation but with high precise timestep, and the synchronised data but without high precise timestep.

Here is the header of the sensor file and ground truth file.

## vicon (vi\*.csv)

Time Header translation.x translation.y translation.z rotation.x rotation.y rotation.z rotation.w

## Sensors (imu\*.csv)

Time attitude_roll(radians) attitude_pitch(radians) attitude_yaw(radians) rotation_rate_x(radians/s) rotation_rate_y(radians/s) rotation_rate_z(radians/s) gravity_x(G) gravity_y(G) gravity_z(G) user_acc_x(G) user_acc_y(G) user_acc_z(G) magnetic_field_x(microteslas) magnetic_field_y(microteslas) magnetic_field_z(microteslas)

## Structure

In this folder

```
user@fedora ~/C/magnetic_localization (master)> ls data/Oxford\ Inertial\ Odometry\ Dataset/handheld/data1/syn/
imu1.csv*  imu4.csv*  imu7.csv*  vi3.csv*  vi6.csv*
imu2.csv*  imu5.csv*  vi1.csv*   vi4.csv*  vi7.csv*
imu3.csv*  imu6.csv*  vi2.csv*   vi5.csv*
user@fedora ~/C/magnetic_localization (master)> ls data/Oxford\ Inertial\ Odometry\ Dataset/handheld/data2/syn/
imu1.csv*  imu2.csv*  imu3.csv*  vi1.csv*  vi2.csv*  vi3.csv*
user@fedora ~/C/magnetic_localization (master)> ls data/Oxford\ Inertial\ Odometry\ Dataset/handheld/
data1/  data3/  data5/          Test.txt*
data2/  data4/  handheld.xlsx*  Train.txt*
user@fedora ~/C/magnetic_localization (master)> pwd
/home/user/Code/magnetic_localization
```

Also like each of them are the same length, so no need to sync the timesteps

```
user@fedora ~/C/magnetic_localization (master)> cat data/Oxford\ Inertial\ Odometry\ Dataset/handheld/data1/syn/imu2.csv |wc
  23446   23446 3282548
user@fedora ~/C/magnetic_localization (master)> cat data/Oxford\ Inertial\ Odometry\ Dataset/handheld/data1/syn/vi2.csv |wc
  23446   23446 1740520
```

Just ignore the Time and Header

## Goal

Our goal is to predict the current `x`, `y`, `z` based on the previous all previous data(but not previous `x`, `y`, `z`)

## Trying Vanilla LSTM

1. Suffers from distribution shift

```
X_train means:
mag_x: -0.46830051313300697
mag_y: -15.668869715403527
mag_z: -36.376663738555756
mag_total: 42.527113564066134

X_test means:
mag_x: -4.326749743812862
mag_y: -13.854210498185887
mag_z: -32.72580701915449
mag_total: 38.786503919304415

y_train means:
x: 0.12568006574318533
y: 0.023374721301369764
z: 1.176188457321701

y_test means:
x: 0.15600621631161352
y: 0.075468841401043
z: 1.1843440265075935
```

2. High variance

```
Epoch [0/49], Train MSE (denorm): 1.4946, Test MSE (denorm): 2.5259
Epoch [0/49], Train Loss: 0.3160
Epoch [1/49], Train MSE (denorm): 0.2647, Test MSE (denorm): 3.6559
Epoch [1/49], Train Loss: 0.2283
Epoch [2/49], Train MSE (denorm): 0.1589, Test MSE (denorm): 3.4956
```

## VIT Approach

### Input Layer

- **Segmentation**: Divide your input sequence into fixed-size "patches" (e.g., 1-second windows).
- **Projection**: Project each patch to a higher-dimensional space using a linear transformation to create patch embeddings.

### Positional Encoding

- **Encoding**: Add learnable or fixed positional embeddings to the patch embeddings to maintain temporal order.

### Transformer Encoder

- **Structure**: Stack of Transformer encoder blocks (e.g., 6-12 layers).
- **Components**:
  - Multi-Head Self-Attention
  - Layer Normalization
  - Feed-Forward Network
  - Residual Connections

### Global Average Pooling

- **Aggregation**: Aggregate information across all patches.

### MLP Head

- **Prediction**: Final layers to predict translation (x, y, z).
