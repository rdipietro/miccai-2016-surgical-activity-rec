#!/bin/bash

export data_dir="~/Data/JIGSAWS/Suturing"

for model_type in BidirectionalLSTM ForwardLSTM
do
    export model_type
    # Each test-users set consists of a single user.
    for test_users in B C D E F G H I
    do
        export test_users
        bash run_one_train_eval.sh "jigsaws $model_type $test_users"
    done
done

