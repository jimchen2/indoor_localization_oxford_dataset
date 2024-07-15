# Getting Started

```
git clone https://github.com/jimchen2/magnetic_localization
```

## AWS EC2 Command to Open up Jupyter(put here ease for copying)

```
sudo su
```

Then

```
apt update && apt upgrade -y
sudo apt install python3 python3-venv -y
cd
python3 -m venv myenv
source myenv/bin/activate
#pip install jupyterlab
jupyter lab --no-browser --allow-root --port 80 --ip 0.0.0.0 # added no-browser
```

## Get data

Go to `https://us-east-1.console.aws.amazon.com/s3` and generate a persigned url like this
`https://jimchen4214-private.s3.us-east-1.amazonaws.com/data.zip?response-content-disposition=.....`, then

```
url="[PRESIGNED_URL]"
wget "$url" -O data.zip && unzip data.zip && rm data.zip
```

# Project Structure


- `prev_scripts/` : All the previous scripts (depreacted)
- `RNN_Framework.ipynb`: Starter and example of what to do
- `saved_models/` : Directory to save models

# Working Together

Open an issue at `https://github.com/jimchen2/magnetic_localization/issues` or create a new branch, anything that works.

# Goal

## What We Have

We have Many of these trajs

```
'x': [160.132662931194, 160.13268418790938, 160.132...], 'y':[130,...], 'Bv':[...], 'Bh':[...],'Bp':[...]

```

Each traj is like this with same number of elements in `x`, `y`, `Bv`, `Bh`, and `Bp`, some trajs have around 700, other have around 5000, and some have other lengths , like phone recording the location and "MAGNETIC" periodically

These trajs neighboring x and y are very near. However, the consecutive difference of `x` and `y` differ (although they are both very small usually), with usually one variable changing and the other staying the same, it is like walking in an almost straight line in a cartesian coordinate system

```
x start 160.132662931194
x end 160.1475851453791
y start 135.33849848577762
y end 113.33606225210056
this traj length 703
x start 269.98628855514494
x end 270.1816159307187
y start 113.46605420976405
y end 135.5595986434642
this traj length 625
x start 128.03530066240282
x end 270.0600066396031
y start 113.34940521938188
y end 113.32196830565285
this traj length 4239
```

We split the data into a training set and a testing set, the testing set containing 9 trajs, with 24589 total tuples, and the training set with 70 trajs and 170013 total tuples

## What to Achieve

The goal is to basically predict the `x` and `y` given a sequence or series of `Bv`, `Bh`, and `Bp`.

# Hooking up `V100` on `AWS EC2`

It's basically the P3-2x instance, quite cheap not gonna lie, except like I basically found myself migrating from ec2-t3 large to this thing so like you need to install the drivers and support for tensorflow!

It's basically [here](https://www.tensorflow.org/install/pip)

## Install nvidia driver

```
sudo apt install nvidia-driver-535 -y
```

verify

```
nvidia-smi
```

Output:

```
Fri Jul 12 06:39:24 2024
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.256.02   Driver Version: 470.256.02   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   41C    P0    25W / 300W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Install nvidia Toolkit

```
sudo apt install nvidia-cuda-toolkit
```

Verify

```
nvcc --version
```

Output

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
```

## Install `cudnn`

```
sudo apt install nvidia-cudnn
```

click through the agreements

then

```
dpkg -l | grep cudnn
```

to verify

```
source myenv/bin/activate
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Basically Rebuild Tensorflow

```ipython
!pip uninstall tensorflow -y
!pip install tensorflow[and-cuda]
```
