cd src/easy_resnet/


python resnet_train.py --sequence_length 200 --input_size 12 --channels 64 128 256 \
    --output_size 3 --learning_rate 0.001 --batch_size 32 --num_epochs 50 \
    --dropout_rate 0.3 --root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket" \
    --test_root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket_test/"

python resnet_train.py --sequence_length 200 --input_size 12 --channels 128 256 512 \
    --output_size 3 --learning_rate 0.001 --batch_size 32 --num_epochs 50 \
    --dropout_rate 0.4 --root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket" \
    --test_root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket_test/"

python resnet_train.py --sequence_length 200 --input_size 12 --channels 64 128 256 \
    --output_size 3 --learning_rate 0.001 --batch_size 32 --num_epochs 50 \
    --dropout_rate 0.5 --root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket" \
    --test_root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket_test/"

python resnet_train.py --sequence_length 200 --input_size 12 --channels 64 128 256 \
    --output_size 3 --learning_rate 0.001 --batch_size 32 --num_epochs 50 \
    --dropout_rate 0.4 --root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket" \
    --test_root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket_test/"




cd ../other_exp/


python lstm_train.py --sequence_length 200 --hidden_sizes 64 --num_epochs 100 \
    --dropout_rate 0.4 --root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket" \
    --test_root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket_test/"

python lstm_train.py --sequence_length 200 --hidden_sizes 128 64 --num_epochs 100 \
    --dropout_rate 0.4 --root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket" \
    --test_root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket_test/"

python lstm_train.py --sequence_length 200 --hidden_sizes 256 128 --num_epochs 100 \
    --dropout_rate 0.4 --root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket" \
    --test_root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket_test/"

python lstm_train.py --sequence_length 200 --hidden_sizes 128 --num_epochs 100 \
    --dropout_rate 0.4 --root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket" \
    --test_root_dir "../../data/Oxford Inertial Odometry Dataset/handheld+handbag+trolley+pocket_test/"


    
cd ../easy_transformer/


python transformer_train.py --d_model 128 --nhead 8 --num_layers 2 --dim_feedforward 1024 \
    --batch_size 32 --num_epochs 50 --dropout_rate 0.3 --pooling last

python transformer_train.py --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256 \
    --batch_size 32 --num_epochs 25 --pooling last

python transformer_train.py --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256 \
    --batch_size 32 --num_epochs 50 --dropout_rate 0.2 --pooling last --return_all_positions