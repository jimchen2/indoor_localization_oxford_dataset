# What This Code Does

## What We Have

We have Many of these trajs

```
'x': [160.132662931194, 160.13268418790938, 160.132...], 'y':[130,...], 'Bv':[...], 'Bh':[...],'Bp':[...]

```

Each traj is like this with same number of elements in `x`, `y`, `Bv`, `Bh`, or `Bp`, some trajs have around 700, other have around 5000, and some have other lengths , like phone recording the location and "MAGNETIC" periodically

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

## Processing the Data

We do some things to preprocess the data before sending it into a sliding window.

But we shouldn't drop too much data when sampling.

## After Processing

After processing(sampling) what we get is many trajs like this, Each consecutive points are neither too far away nor too close. (one of the `x`,`y` are not moving much since it's a corridor)

```
{'x': [160.134, 160.132, 160.133...], 'y':[130.1,130.5,130.8...], 'Bv':[...], 'Bh':[...],'Bp':[...]}
```

or

```
{'x': [204.1, 204.5, 204.8...], 'y':[113.349,113.349,113.348...], 'Bv':[...], 'Bh':[...],'Bp':[...]}
```

We can shuffle it (it doesn't hurt anyway) here.

## Applying Window(Deterministic Process)

So basically for each traj we get a series of `Bv`, `Bh`, `Bp`, akin to real world scenario, following up to one pair of `x` and `y` (which we are trying to predict)

## Other Options(hyperparameters for tuning)

1. Flipping Data
2. Applying Random Noises(Only to training set)
3. Window Size
4. Batch Size
5. Epoches
6. Patience
7. Partition_Of_DataSets

## Model Hyperparameters

We have a reference model(might not work) and it can be finetuned, note that we should add patience to avoid overfitting

```
model = keras.models.Sequential([
        keras.layers.BatchNormalization(input_shape=(WINDOW_SIZE, 3)),  # Add this line
        TCN(input_shape=(WINDOW_SIZE, 3), nb_filters=8, return_sequences=True, dilations=[1, 2, 4, 8, 16]),
        # keras.layers.LSTM(units=8, input_shape=(10, 8), return_sequences=False),
        keras.layers.Flatten(),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.1),
        keras.layers.Dense(16),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.1),
        keras.layers.Dense(8),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.1),
        keras.layers.Dense(4),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.1),
        keras.layers.Dense(2)
    ])
model.compile(optimizer='adam', loss='mae', metrics=[keras.metrics.MeanAbsoluteError()])
```

## After Programming

1. Do the same training for each config 5 times (don't get average, just check to see the eval differences then log)
2. Choose different models
3. Choose different hyperparams

## What to Log to TensorBoard

The model evaluation on training and validation each time and the final test set model loss and the total epoches

## Thought-Provoking Questions:

**Assumptions**: We have enough compute gpus

1. How do we preprocess the data? Because in real world you are only looking at a history of magnetic fields to predict the current `x` and `y` with a sliding window here, and in the training you are essentially looking at previous `x` and `y`, so it's a complete different story.
2. Should we divide the validation datasets?(I think that is a good approach since that separates the validation and the training datasets, the test dataset is already separated, essentially we can do 8-1-1 here)
3. Since the data is very limited, is it a good idea to apply to the same data with different noises to train on? Or since we have enough computing power in this scenario, and adding noises won't give the model more insight to the data itself, maybe we should apply noise only once?
4. Should we use k-fold?(but note that we generally create several trajs on each specific occasion, and those trajs are very similar, so if we are using k-fold, should we make adjustments? Read More: Leave-One-Group-Out)

## JC's Thoughts

1. The test dataset will go through both being processed and being evaluated for us to get an idea. In real world the `x` and `y` aren't sampled to be a fixed length, instead, they are unknown. In training we are not supposed to have any information about the testing datasets `x` and `y`. So, we partition each traj like 4th, 4th+1, 4th+2 point...(My idea for simplicity, it can also be finetuned, we may also sample the test data like this)
2. Yes, We should divide it to validation datasets. Prevent data leakage.
3. No, We don't apply noise multiple times as it's not useful. Generating more useless data won't provide additional insights.(It might lead to faster convergence, which doesn't apply as we have enough compute)
4. To be answered(we do everything else first then if we are too tired we give up)
