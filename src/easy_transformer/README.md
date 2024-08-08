## Easy Transformer

Transformer is more complicated to implement than RNN, and thus very prune to mistakes. I am workng hard to make sure no misconfigs happen(though it is very likely).

## Run Some Configuarations

So basically run different configs to get an idea what happens


```bash
# Make sure no bugs
python transformer_train.py --num_epochs 10

# Baseline configuration
python transformer_train.py --d_model 32 --nhead 2 --num_layers 1 --dim_feedforward 128 --batch_size 64 --num_epochs 50 --pooling last

# Deeper model
python transformer_train.py --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256 --batch_size 32 --num_epochs 50 --pooling last
python transformer_train.py --d_model 64 --nhead 4 --num_layers 6 --dim_feedforward 256 --batch_size 32 --num_epochs 75 --pooling last

# Wider model
python transformer_train.py --d_model 128 --nhead 8 --num_layers 2 --dim_feedforward 1024 --batch_size 16 --num_epochs 50 --dropout_rate 0.3 --pooling last

# Mean pooling
python transformer_train.py --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256 --batch_size 32 --num_epochs 50 --dropout_rate 0.2 --pooling mean

# Return all positions
python transformer_train.py --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256 --batch_size 32 --num_epochs 50 --dropout_rate 0.2 --pooling last --return_all_positions

# Larger batch size
python transformer_train.py --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256 --batch_size 128 --num_epochs 30 --dropout_rate 0.2 --pooling last

# More training epochs
python transformer_train.py --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256 --batch_size 32 --num_epochs 100 --dropout_rate 0.2 --pooling last

# Higher dropout
python transformer_train.py --d_model 64 --nhead 4 --num_layers 3 --dim_feedforward 512 --batch_size 32 --num_epochs 75 --dropout_rate 0.5 --pooling last
```
