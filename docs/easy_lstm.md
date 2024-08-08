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
Epoch [1/50], Train Loss: 1.0333, Val Loss: 0.7039
Epoch [2/50], Train Loss: 0.6933, Val Loss: 0.7423
Epoch [3/50], Train Loss: 0.5872, Val Loss: 0.6355
Epoch [4/50], Train Loss: 0.5469, Val Loss: 0.5512
Epoch [5/50], Train Loss: 0.6917, Val Loss: 0.6491
Epoch [6/50], Train Loss: 0.5254, Val Loss: 0.5288
Epoch [7/50], Train Loss: 0.4866, Val Loss: 0.4836
Epoch [8/50], Train Loss: 0.4751, Val Loss: 0.5284
Epoch [9/50], Train Loss: 0.4397, Val Loss: 0.5155
Epoch [10/50], Train Loss: 0.5113, Val Loss: 0.5460
Epoch [11/50], Train Loss: 0.4426, Val Loss: 0.4561
Epoch [12/50], Train Loss: 0.4397, Val Loss: 0.5362
Epoch [13/50], Train Loss: 0.4454, Val Loss: 0.6536
Epoch [14/50], Train Loss: 0.4092, Val Loss: 0.4735
Epoch [15/50], Train Loss: 0.3930, Val Loss: 0.4855
Epoch [16/50], Train Loss: 0.3816, Val Loss: 0.3979
Epoch [17/50], Train Loss: 0.3616, Val Loss: 0.4172
Epoch [18/50], Train Loss: 0.3498, Val Loss: 0.4247
Epoch [19/50], Train Loss: 0.3768, Val Loss: 0.3786
Epoch [20/50], Train Loss: 0.3466, Val Loss: 0.3950
Epoch [21/50], Train Loss: 0.3492, Val Loss: 0.4395
Epoch [22/50], Train Loss: 0.3466, Val Loss: 0.4699
Epoch [23/50], Train Loss: 0.3274, Val Loss: 0.4523
Epoch [24/50], Train Loss: 0.3463, Val Loss: 0.3629
Epoch [25/50], Train Loss: 0.3294, Val Loss: 0.4008
Epoch [26/50], Train Loss: 0.3051, Val Loss: 0.3653
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
Epoch [1/50], Train Loss: 1.3684, Val Loss: 0.8679
Epoch [2/50], Train Loss: 0.7621, Val Loss: 0.6187
Epoch [3/50], Train Loss: 0.6352, Val Loss: 0.5016
Epoch [4/50], Train Loss: 0.5431, Val Loss: 0.5001
Epoch [5/50], Train Loss: 0.5197, Val Loss: 0.4581
Epoch [6/50], Train Loss: 0.4790, Val Loss: 0.4571
Epoch [7/50], Train Loss: 0.4949, Val Loss: 0.4723
Epoch [8/50], Train Loss: 0.4803, Val Loss: 0.4327
Epoch [9/50], Train Loss: 0.4594, Val Loss: 0.5488
Epoch [10/50], Train Loss: 0.5856, Val Loss: 0.4645
Epoch [11/50], Train Loss: 0.4688, Val Loss: 0.4248
Epoch [12/50], Train Loss: 0.4309, Val Loss: 0.4892
Epoch [13/50], Train Loss: 0.4448, Val Loss: 0.4438
Epoch [14/50], Train Loss: 0.4329, Val Loss: 0.4275
Epoch [15/50], Train Loss: 0.4282, Val Loss: 0.4330
Epoch [16/50], Train Loss: 0.3844, Val Loss: 0.4155
Epoch [17/50], Train Loss: 0.4858, Val Loss: 0.4628
Epoch [18/50], Train Loss: 0.4316, Val Loss: 0.4598
Epoch [19/50], Train Loss: 0.3948, Val Loss: 0.4880
Epoch [20/50], Train Loss: 0.4565, Val Loss: 0.4461
Epoch [21/50], Train Loss: 0.3910, Val Loss: 0.4228
Epoch [22/50], Train Loss: 0.3865, Val Loss: 0.4076
Epoch [23/50], Train Loss: 0.3608, Val Loss: 0.3852
Epoch [24/50], Train Loss: 0.3708, Val Loss: 0.3919
Epoch [25/50], Train Loss: 0.3967, Val Loss: 0.3775
Epoch [26/50], Train Loss: 0.3489, Val Loss: 0.3928
Epoch [27/50], Train Loss: 0.3337, Val Loss: 0.3806
Epoch [28/50], Train Loss: 0.3433, Val Loss: 0.4663
Epoch [29/50], Train Loss: 0.3408, Val Loss: 0.3440
```