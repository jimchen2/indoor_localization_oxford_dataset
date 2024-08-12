## Easy Transformer

Transformer is more complicated to implement than RNN, and thus very prune to mistakes. I am workng hard to make sure no misconfigs happen(though it is very likely).


## Performance

It's like transfomer generally is better than LSTM(at least in my implementation), but not by a lot, it suffers much worse from overfitting.

## Run Some Configuarations

So basically run different configs to get an idea what happens



```bash
# Baseline configuration
python transformer_train.py --d_model 32 --nhead 2 --num_layers 1 --dim_feedforward 128 --batch_size 64 --num_epochs 50 --pooling last

# Deeper model
python transformer_train.py --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256 --batch_size 32 --num_epochs 25 --pooling last
python transformer_train.py --d_model 64 --nhead 4 --num_layers 6 --dim_feedforward 256 --batch_size 32 --num_epochs 25 --pooling last

# Wider model
python transformer_train.py --d_model 128 --nhead 8 --num_layers 2 --dim_feedforward 1024 --batch_size 16 --num_epochs 50 --dropout_rate 0.2 --pooling last
python transformer_train.py --d_model 128 --nhead 8 --num_layers 2 --dim_feedforward 1024 --batch_size 16 --num_epochs 50 --dropout_rate 0.3 --pooling last

# Mean pooling
python transformer_train.py --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256 --batch_size 32 --num_epochs 50 --dropout_rate 0.2 --pooling mean

# Return all positions
python transformer_train.py --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256 --batch_size 32 --num_epochs 50 --dropout_rate 0.2 --pooling last --return_all_positions

# Larger batch size
python transformer_train.py --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256 --batch_size 128 --num_epochs 30 --dropout_rate 0.2 --pooling last

# Higher dropout
python transformer_train.py --d_model 64 --nhead 4 --num_layers 3 --dim_feedforward 512 --batch_size 32 --num_epochs 75 --dropout_rate 0.5 --pooling last
```

## Results

![Image](https://blog.jimchen.me/d68c4bf0-2fde-4856-8041-a98f49130801-1723479132586.jpg)

![Image](https://blog.jimchen.me/dfed22dd-14f9-4403-bd56-7837655ffced-1723479150777.jpg)

![Image](https://blog.jimchen.me/72cc5514-bb10-4b41-a224-722fb9a65873-1723479160425.jpg)

![Image](https://blog.jimchen.me/8cfae264-dcc4-48c4-8362-bfd4aedadaf5-1723479166719.jpg)

## Observations

Transformer easily achieves a much better baseline but seems to be an overkill in this problem with limited data. It have a huge variance and the result is on par with LSTM.