# All Notes about the project!

# Useful Commands (Pinned to Front)

## Clone the debloated branch

```
git clone -b JC_branch --single-branch https://github.com/mtrec-magnetic-localization/sequence-model-trial.git
```

## AWS EC2 Command to Open up Jupyter(put here ease for copying)

```
sudo su
apt update && apt upgrade -y

sudo apt install python3 python3-venv -y
cd ~
python3 -m venv myenv
source myenv/bin/activate
#pip install jupyterlab
jupyter lab --no-browser --allow-root --port 80 --ip 0.0.0.0 # added no-browser
```

## Put the data folder on the S3

Never Bloat the Github Repo! We can basically we can get it on the fly

Go to `https://us-east-1.console.aws.amazon.com/s3` and generate a persigned url like this
`https://jimchen4214-private.s3.us-east-1.amazonaws.com/data.zip?response-content-disposition=.....`, then

```
url="[PRESIGNED_URL]"
wget "$url" -O data.zip && unzip data.zip && rm data.zip
```

# Table of Contents

1. [Useful Commands (Pinned to Front)](#useful-commands-pinned-to-front)

   - [Clone the debloated branch](#clone-the-debloated-branch)
   - [AWS EC2 Command to Open up Jupyter](#aws-ec2-command-to-open-up-jupyterput-here-ease-for-copying)
   - [Put the data folder on the S3](#put-the-data-folder-on-the-s3)

2. [Updates 2024.7.13](#updates-2024713)

   - [Removed the preprocessing of data](#removed-the-preprocessing-of-data)
   - [The Validation and Training Datasets are mixed together](#the-validation-and-training-datasets-are-mixed-together)
   - [Problems](#problems)
   - [Removed the SAMPLE_ITERATION](#removed-the-sample_iteration)
   - [Data leakage between train/valid datasets](#there-is-data-leakage-between-trainvalid-datasets-we-are-also-going-to-fix-that)
   - [Don't add noise to valid datasets](#dont-add-noise-to-valid-datasets)

3. [Updates 2024.7.12](#updates-2024712)

   - [Made function `sample_data` and `random_speed_sample` conciser](#made-function-sample_data-and-random_speed_sample-conciser)
   - [Tried and Benchmarked Some Results](#tried-and-benchmarked-some-resultstodo)
   - [Put the parameters forward and argparse them](#basically-put-the-parameters-forward-and-argparse-thembuggy-debug-tomorrow)
   - [Converted it to python](#converted-it-to-python-buggy-debug-tomorrow)
   - [Added tensorboard for logging](#added-tensorboard-for-logging-buggy-debug-tomorrow)

4. [Updates 2024.7.11](#updates-2024711)

   - [Hooking up V100 on AWS EC2](#hooking-up-v100-on-aws-ec2)
   - [Install nvidia driver](#install-nvidia-driver)
   - [Install nvidia Toolkit](#install-nvidia-toolkit)
   - [Install cudnn](#install-cudnn)
   - [Basically Rebuild Tensorflow](#basically-rebuild-tensorflow)
   - [Add a minus sign in the code](#add-a-minus-sign-in-the-code)

5. [Updates 2024.7.10](#updates-2024710)

# Updates 2024.7.14

## Implements Training on Subsets

First train on subset for the loss to go down and quickly

## Changed the Window Iterator

WTFFFF? THis is so freaking funny, like the window was previously iterating through the keys(taking the fisrst 5 window what the FREAK???)

So I Fixed the stupid error in `process_dataset`

Mistaken of variables, stupid mistake

## Put HyperParams in front

Ready to log to tensorboard

## Added the Dataset Partition Method

## Added the Other Noises other than Gaussian Random Noise

## Fixed bugs in evaluation of Training

# Updates 2024.7.13

## Removed the preprocessing of data

The test dataset will go through both being processed and being evaluated for us to get an idea. In real world the `x` and `y` aren't sampled to be a fixed length, instead, they are unknown. In training we are not supposed to have any information about the testing datasets `x` and `y`. So, we partition each traj like 4th, 4th+1, 4th+2 point...(My idea for simplicity, it can also be finetuned)

## The Validation and Training Datasets are mixed together

BRUHHHHH???????????

_Anyways_, fixed [x]

## Problems

I am trying to solve them

1. Training Loss is always much bigger than Validation Loss?????
2. Why is test dataset in a seperate folder, not split 7-2-1???

## Removed the SAMPLE_ITERATION

So basically in the original code there is this snippet(something like this)

```
for count in range(SAMPLE_ITERATION):
    for d in train_raw_data:
        for window_idx in range(0, len(d)):
            window_start = window_idx
            window_end = window_idx + WINDOW_SIZE
```

We don't need to sample multiple times as the code runs quickly on low end cpus(just increase the iter)

## There is data leakage between train/valid datasets we are also going to fix that

## Don't add noise to valid datasets

# Updates 2024.7.12

The code is quite verbose unfortunately, so I am doing a few changes to the ipynb for ease of use

## Made function `sample_data` and `random_speed_sample` conciser

## Tried and Benchmarked Some Results(todo)

See the report pdf(todo)

## Basically put the parameters forward and argparse them(buggy, debug tomorrow)

## Converted it to python (buggy, debug tomorrow)

## Added tensorboard for logging (buggy, debug tomorrow)

# Updates 2024.7.11

## Hooking up `V100` on `AWS EC2`

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

## Add a minus sign in the code

```
y = - float(row[4]) / PIXEL_TO_METER_SCALE + image.size[1]
```

# Updates 2024.7.10

1. Added `requirements.txt` for specific versions
2. Changed the backslashes(probably in Window$ to `os.join`), the `os.join` is cross-platform compatible
3. Changed a few things in TCN which didn't work previously, (probably due to version differences), like updated this line

```
model.compile(optimizer='adam', loss='mae', metrics=[keras.metrics.MeanAbsoluteError()])
```

4. A suggestion: It might be beneficial to move the larger files in `/data/raw` (1 GB+) out of the Git repository. These files could be stored separately on Amazon S3 and downloaded on the fly(or mount the S3 bucket if needed on AWS EC2/SageMaker). This approach is sometimes used in ML projects to keep repositories lightweight.

There is a limit on how large a git repo can be I think. (or perhaps use git-lfs and put them on huggingface?)

5. The `h5` is considered legacy so maybe we can save it as

```
model.save('saved_model/4F_survey_data_diff_speed_5.32m.keras')
```

With these modifications the code will be able to run on AWS EC2 t3a instances.
