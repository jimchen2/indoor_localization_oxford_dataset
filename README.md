## Setup

First wget https://s3.amazonaws.com...  (removed due to copyright issues)

In each data fold, there is a raw data subfolder and a syn data subfolder, which represent the raw data collection without synchronisation but with high precise timestep, and the synchronised data but without high precise timestep.


## In Google Colab


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimchen2/indoor_localization_oxford_dataset/blob/master/localization_run.ipynb)


**Here I also changed the Oxford Datasets, so I seperated the Oxford into 2 folders for train and test instead of putting the information in each folder.**

Here is the header of the sensor file and ground truth file.

In each data fold, there is a raw data subfolder and a syn data subfolder, which represent the raw data collection without synchronisation but with high precise timestep, and the synchronised data but without high precise timestep.

Here is the header of the sensor file and ground truth file.

## vicon (vi*.csv)

Time  Header  translation.x translation.y translation.z rotation.x rotation.y rotation.z rotation.w

## Sensors (imu*.csv)

Time attitude_roll(radians) attitude_pitch(radians) attitude_yaw(radians) rotation_rate_x(radians/s) rotation_rate_y(radians/s) rotation_rate_z(radians/s) gravity_x(G) gravity_y(G) gravity_z(G) user_acc_x(G) user_acc_y(G) user_acc_z(G) magnetic_field_x(microteslas) magnetic_field_y(microteslas) magnetic_field_z(microteslas)

## Structure

In this folder
```
user@fedora ~/C/magnetic_localization (master)> ls data/Oxford\ Inertial\ Odometry\ Dataset/handheld/
data1/  data3/  data5/          Test.txt*
data2/  data4/  handheld.xlsx*  Train.txt*
user@fedora ~/C/magnetic_localization (master)> ls data/Oxford\ Inertial\ Odometry\ Dataset/handheld/data1/syn/
imu1.csv*  imu4.csv*  imu7.csv*  vi3.csv*  vi6.csv*
imu2.csv*  imu5.csv*  vi1.csv*   vi4.csv*  vi7.csv*
imu3.csv*  imu6.csv*  vi2.csv*   vi5.csv*
user@fedora ~/C/magnetic_localization (master)> ls data/Oxford\ Inertial\ Odometry\ Dataset/handheld/data2/syn/
imu1.csv*  imu2.csv*  imu3.csv*  vi1.csv*  vi2.csv*  vi3.csv*
user@fedora ~/C/magnetic_localization (master)> ls data/Oxford\ Inertial\ Odometry\ Dataset/handheld/data3/syn/
imu1.csv*  imu3.csv*  imu5.csv*  vi2.csv*  vi4.csv*
imu2.csv*  imu4.csv*  vi1.csv*   vi3.csv*  vi5.csv*
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

Just fucking ignore the Time and Header

## Some more Information
```
user@fedora ~/C/m/d/O/h/d/syn (master)> pwd
/home/user/Code/magnetic_localization/data/Oxford Inertial Odometry Dataset/handheld/data1/syn
user@fedora ~/C/m/d/O/h/d/syn (master)> ls
imu1.csv*  imu4.csv*  imu7.csv*  vi3.csv*  vi6.csv*
imu2.csv*  imu5.csv*  vi1.csv*   vi4.csv*  vi7.csv*
imu3.csv*  imu6.csv*  vi2.csv*   vi5.csv*
user@fedora ~/C/m/d/O/h/d/syn (master)> cat imu1.csv |head -n 5
1.50E+11,0.003649,0.44925,-0.21255,0.036483,-0.029496,0.020632,0.003286,-0.43429,-0.90077,-0.002798,0.014599,-0.016836,-2.3577,-0.3717,-42.347
1.50E+11,0.00305,0.44954,-0.21219,0.067307,-0.038284,0.029241,0.002747,-0.43455,-0.90064,-0.004578,0.013712,-0.013968,-2.3576,-0.37186,-42.272
1.50E+11,0.002363,0.45003,-0.21184,0.076935,-0.039423,0.021788,0.002128,-0.43499,-0.90043,-0.007743,0.013192,-0.008427,-2.3576,-0.37186,-42.272
1.50E+11,0.001778,0.45053,-0.21167,0.066339,-0.039321,0.006826,0.001601,-0.43544,-0.90021,-0.006255,0.011814,-0.003259,-2.207,-0.59618,-42.27
1.50E+11,0.001393,0.45086,-0.21171,0.037685,-0.030543,-0.010317,0.001253,-0.43574,-0.90007,-0.003634,0.008648,0.000367,-2.207,-0.59618,-42.27
user@fedora ~/C/m/d/O/h/d/syn (master)> cat vi1.csv |head -n 5
1.50E+11,12978,-1.2991,1.7212,1.1931,-0.21409,-0.012459,-0.097183,0.97189
1.50E+11,12979,-1.2993,1.7213,1.1932,-0.21473,-0.01237,-0.097239,0.97174
1.50E+11,12980,-1.2993,1.7213,1.1932,-0.21509,-0.012288,-0.097244,0.97166
1.50E+11,12981,-1.2994,1.7214,1.1933,-0.21526,-0.012291,-0.09738,0.97161
1.50E+11,12982,-1.2995,1.7213,1.1933,-0.21541,-0.012199,-0.097503,0.97157
```

Note: Use os list dir instead of hardcoding data folders

# **YOU MUST SPLIT DATA AT THE FILE LEVEL TO PREVENT LEAKAGE**

## TENSORBOARD Logs

Tensorboard logs shouldn't be put in Git, so I put them here

`https://jimchen4214-public.s3.amazonaws.com/other/mag_tensorboard_logs/logs.zip`
`https://jimchen4214-public.s3.amazonaws.com/other/mag_tensorboard_logs/lstm_logs.zip`


Command to upload
```bash
zip -r logs.zip logs/
aws s3 cp logs.zip s3://jimchen4214-public/other/mag_tensorboard_logs/logs.zip
```

## Goal

Our goal is to predict the current `x`, `y`, `z` based on the previous all previous data(but not previous `x`, `y`, `z`)

## Trying Vanilla LSTM without splitting windows

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


## A list of things I am not doing

1. I am not implementing variants of LSTM or RNN because I don't think it's that meaningful, you can do it if you want
2. I don't think pure TCN is going to work. Maybe utilizing parallel CNNs for input layer might be a good idea.
3. I am not changing the `Bx`, `By` `Bz` into other forms because I think they yield similar results with all these models(the models automatically fits it), you can do it if you want
4. I am using the layernorm in the start of the model instead of scaling it manually


# **Improvements**

## Variance Too High

If we use the sliding window approach we can easily like make sure they don't overlap to make variance much smaller.

## Sequence Too Short

If the sequence is too short then it performs poorly, improving the sequence length to 200 or 300 drastically improves the performance.

## Remove Parts of IMU

Removing the attitudes make the result better a little bit. Removing the derivation of attitudes(rotation rate) however, makes the result much worse.

## More Epochs

Somehow if I use more epochs it still improve a lot? Although the Val Loss isn't decreasing by a lot though.

