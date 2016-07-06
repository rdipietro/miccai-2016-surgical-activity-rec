#!/bin/bash

data_dir="~/Data/MISTIC"
model_type="BidirectionalLSTM"
test_users="03 04 10 30"

for num_layers in 1 2
do
    for hidden_layer_size in 64 128 256 512 1024
    do
        for dropout_keep_prob in 0.5 1.0
        do
            sbatch --time=0-16:0:0 run_one_job.sh \
                --data_dir "$data_dir" \
                --model_type "$model_type" \
                --test_users "$test_users" \
                --num_layers "$num_layers" \
                --hidden_layer_size "$hidden_layer_size" \
                --dropout_keep_prob "$dropout_keep_prob"
        done
    done
done
