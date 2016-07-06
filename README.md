# Recognizing Surgical Activities with Recurrent Neural Networks

This is the accompanying code for R. DiPietro, C. Lea, A. Malpani,
N. Ahmidi, S.S. Vedula, G.I. Lee, M.R. Lee, G.D. Hager: Recognizing Surgical
Activities with Recurrent Neural Networks. Medical Image Computing
and Computer Assisted Intervention (MICCAI), 2016.

[http://arxiv.org/abs/1606.06329](http://arxiv.org/abs/1606.06329)

## Overview

This code builds, trains, and evaluates recurrent neural networks that map
surgical-robot kinematics over time (for example left-hand position,
right-hand position) to surgical activities over time (for example grasping a
needle or tying a knot). We train and evaluate using two datasets, JIGSAWS and
MISTIC-SL. Please see the paper above for more information.

`standardize_jigsaws.py` can be executed to standardize the JIGSAWS dataset
(using the raw JIGSAWS download described below).

`train_and_summarize.py` can be executed to train and evaluate using one test
setup and one set of parameters. (Optional command-line arguments can be
provided for other test setups or parameters; see below.)

`scripts/` contains bash and slurm scripts for batch runs, both for
hyperparameter selection (`run_mistic_validation_batch.sh`) and to compute
results in a leave-one-user-out fashion (`run_jigsaws_users_batch.sh` and
`run_mistic_users_batch.sh`). These scripts are tailored to our slurm setup,
but for reference we also include the bash script
`run_jigsaws_users_batch_sequential.sh`, which simply runs jobs sequentially.

## Running the Code

### Requirements

This code is intended to be run with Python 2.7. You'll also need

- NumPy. `pip install numpy`
- matplotlib. `pip install matplotlib`
- TensorFlow. [https://www.tensorflow.org/](https://www.tensorflow.org/).
You'll need 0.8 or newer (which provides `tf.scan`).

### Obtaining the Data

The JIGSAWS dataset is available
[here](http://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/). After
registering, an automated system will send you a download link. To run this
code, you'll need to download the Suturing Kinematics data.

The default data directory for JIGSAWS is `~/Data/JIGSAWS/Suturing`, so that,
for example, `~/Data/JIGSAWS/Suturing/kinematics` exists. If you put the data
somewhere else, you'll need to specify the location using the `--data_dir`
command-line option.

It is not possible for us to release the MISTIC-SL dataset at this time,
though we hope we will be able to release it in the future.

### Cloning this Repository

```
git clone https://github.com/rdipietro/miccai-2016-surgical-activity-rec.git
```

### Standardizing the Data

To process JIGSAWS and MISTIC-SL using the same code, we first standardize the
data.

If you put the raw JIGSAWS data in the default location, you can standardize
it by running `python standardize_jigsaws.py`. Otherwise you'll need to specify
the data directory:

```
$ python standardize_jigsaws.py -h
usage: standardize_jigsaws.py [-h] [--data_dir DATA_DIR]
                              [--data_filename DATA_FILENAME]

Create a standardized data file from raw data.

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Data directory. (default: ~/Data/JIGSAWS/Suturing)
  --data_filename DATA_FILENAME
                        The name of the standardized-data pkl file that we'll
                        create inside data_dir. (default:
                        standardized_data.pkl)
```

### Training and Evaluating

You can run `python train_and_summarize.py` to train and export evaluation
summaries using the default test setup and default parameters. The default
configuration trains and evaluates using JIGSAWS, with the first user (B) being
held out for testing. You can change all defaults with optional command-line
arguments:

```
$ python train_and_summarize.py -h
usage: train_and_summarize.py [-h] [--data_dir DATA_DIR]
                              [--data_filename DATA_FILENAME]
                              [--test_users TEST_USERS]
                              [--model_type MODEL_TYPE]
                              [--num_layers NUM_LAYERS]
                              [--hidden_layer_size HIDDEN_LAYER_SIZE]
                              [--dropout_keep_prob DROPOUT_KEEP_PROB]
                              [--batch_size BATCH_SIZE]
                              [--num_train_sweeps NUM_TRAIN_SWEEPS]
                              [--initial_learning_rate INITIAL_LEARNING_RATE]
                              [--num_initial_sweeps NUM_INITIAL_SWEEPS]
                              [--num_sweeps_per_decay NUM_SWEEPS_PER_DECAY]
                              [--decay_factor DECAY_FACTOR]
                              [--max_global_grad_norm MAX_GLOBAL_GRAD_NORM]
                              [--init_scale INIT_SCALE]
                              [--num_sweeps_per_summary NUM_SWEEPS_PER_SUMMARY]
                              [--num_sweeps_per_save NUM_SWEEPS_PER_SAVE]

Run training and export summaries to data_dir/logs for a single test setup and
a single set of parameters. Summaries include a) TensorBoard summaries, b) the
latest train/test accuracies and raw edit distances (status.txt), c) the
latest test predictions along with test ground-truth labels
(test_label_seqs.pkl, test_prediction_seqs.pkl), d) visualizations as training
progresses (test_visualizations_######.png).

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Data directory. (default: ~/Data/JIGSAWS/Suturing)
  --data_filename DATA_FILENAME
                        The name of the standardized-data pkl file that
                        resides in data_dir. (default: standardized_data.pkl)
  --test_users TEST_USERS
                        A string of the users that make up the test set, with
                        users separated by spaces. (default: B)
  --model_type MODEL_TYPE
                        The model type, either BidirectionalLSTM, ForwardLSTM,
                        or ReverseLSTM. (default: BidirectionalLSTM)
  --num_layers NUM_LAYERS
                        The number of hidden layers. (default: 1)
  --hidden_layer_size HIDDEN_LAYER_SIZE
                        The number of hidden units per layer. (default: 1024)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        The fraction of inputs to keep whenever dropout is
                        applied. (default: 0.5)
  --batch_size BATCH_SIZE
                        The number of sequences in a batch/sweep. (default: 5)
  --num_train_sweeps NUM_TRAIN_SWEEPS
                        The number of training sweeps. A sweep is a collection
                        of batch_size sequences that continue together
                        throughout time until all sequences in the batch are
                        exhausted. Short sequences grow by being wrapped
                        around in time. (default: 600)
  --initial_learning_rate INITIAL_LEARNING_RATE
                        The initial learning rate. (default: 1.0)
  --num_initial_sweeps NUM_INITIAL_SWEEPS
                        The number of initial sweeps before the learning rate
                        begins to decay. (default: 300)
  --num_sweeps_per_decay NUM_SWEEPS_PER_DECAY
                        The number of sweeps per learning-rate decay, once
                        decaying begins. (default: 50)
  --decay_factor DECAY_FACTOR
                        The multiplicative learning-rate-decay factor.
                        (default: 0.5)
  --max_global_grad_norm MAX_GLOBAL_GRAD_NORM
                        The global norm is the norm of all gradients when
                        concatenated together. If this global norm exceeds
                        max_global_grad_norm, then all gradients are rescaled
                        so that the global norm becomes max_global_grad_norm.
                        (default: 1.0)
  --init_scale INIT_SCALE
                        All weights will be initialized using a uniform
                        distribution over [-init_scale, init_scale]. (default:
                        0.1)
  --num_sweeps_per_summary NUM_SWEEPS_PER_SUMMARY
                        The number of sweeps between summaries. Note: 7 sweeps
                        with 5 sequences per sweep corresponds to (more than)
                        35 visited sequences, which is approximately 1 epoch.
                        (default: 7)
  --num_sweeps_per_save NUM_SWEEPS_PER_SAVE
                        The number of sweeps between saves. (default: 7)
```

## Additional Notes

### JIGSAWS Training Time

You might notice that JIGSAWS trains extremely quickly, and that the
number of training sweeps can be drastically reduced. This is simply because
*all* hyperparameters were chosen using the MISTIC-SL validation set, including
the learning-rate schedule. Here is a relevant snippet from the paper:

```
Because JIGSAWS has a fixed leave-one-user-out test setup, with all users
appearing in the test set exactly once, it is not possible to use JIGSAWS for
hyperparameter selection without inadvertently training on the test set. We
therefore choose all hyperparameters using a small MISTIC-SL validation set
consisting of 4 users (those with only one trial each), and we use the resulting
hyperparameters for both JIGSAWS experiments and MISTIC-SL experiments.
```

### 7/6/2016: Code Refactored

The code has been refactored to be more modular and to have
fewer dependencies (IPython and Jupyter are no longer required). The
refactored code was tested by reproducing the results in the paper. If you
prefer the old version, containing Jupyter notebooks instead of Python files,
then run `git checkout notebooks` after cloning the repository.
