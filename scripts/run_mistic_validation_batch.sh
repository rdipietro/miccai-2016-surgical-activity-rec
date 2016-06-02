#!/bin/bash

export data_dir="~/Data/MISTIC"
export model_type="BidirectionalLSTM"
export test_users="03 04 10 30"

for num_layers in 1 2
do
    export num_layers
    for hidden_layer_size in 64 128 256 512 1024
    do
        export hidden_layer_size
        for dropout_keep_prob in 0.5 1.0
        do
            export dropout_keep_prob
            sbatch run_one_train_eval.sh "val $num_layers $hidden_layer_size $dropout_keep_prob"
        done
    done
done

