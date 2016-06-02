# Recognizing Surgical Activities with Recurrent Neural Networks

This is the accompanying code for R. DiPietro, C. Lea, A. Malpani,
N. Ahmidi, S. Vedula, G.I. Lee, M.R. Lee, G.D. Hager: Recognizing Surgical
Activities with Recurrent Neural Networks. Medical Image Computing
and Computer Assisted Intervention (MICCAI), 2016.

## Overview

This code builds, trains, and evaluates recurrent neural networks that map
robot kinematics to surgical activities: the RNNs take a sequence of
kinematic signals as input and produce a corresponding sequence of
surgical-activity labels as output.

Overview of files:

- `standardize_jigsaws.ipynb`. This Jupyter notebook is used to convert the
raw JIGSAWS dataset into a standardized format, and to summarize the dataset
along the way.
- `train_and_eval.ipynb`. This Jupyter notebook contains the code's core, from
normalizing input data to defining, training, and evaluating the RNNs.
- `scripts/`. This directory contains bash/slurm scripts for batch runs. The
slurm scripts are of course customized for our setup, but
`run_jigsaws_users_batch_sequential.sh` is included and should run on any
machine, though it will run jobs sequentially and therefore slowly.

## Running the Code

### Requirements

This code is intended to be run with Python 2.7. You'll also need

- IPython. `pip install ipython`
- NumPy. `pip install numpy`
- matplotlib. `pip install matplotlib`
- Jupyter. `pip install jupyter`
- TensorFlow. [https://www.tensorflow.org/](https://www.tensorflow.org/).
You'll need 0.8 or newer (which provides `tf.scan`).

### Obtaining the Data

The JIGSAWS dataset is available
[here](http://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/). After
registering, an automated system will send you a download link. To run this
code, you'll need at least the Suturing Kinematics data.

If you put the data in `~/Data/JIGSAWS/Suturing` so that (for example) the
directory `~/Data/JIGSAWS/Suturing/kinematics` exists, then everything should
just work. Otherwise, you'll need to modify the directories/paths in the Jupyter
notebooks. (Or, if you need to use a different disk etc., a symbolic link in
`~/Data` will of course work too.)

### Cloning this Repository

```
git clone https://github.com/rdipietro/miccai-2016-surgical-activity-rec.git
```

### Running Jupyter

```
jupyter notebook
```

Navigate to http://localhost:8888

### Standardizing the Data

Open and run `standardize_jigsaws.ipynb` using Jupyter.

This will populate the notebook with information and plots that summarize the
JIGSAWS dataset, and it will also create a standardized data file,
`standardized_data.pkl`, in the JIGSAWS data directory. This standardized format
was used for convenience so that JIGSAWS and MISTIC could be subsequently
processed in the same way.

### Training and Evaluating

Open `train_and_eval.ipynb`.

This is the core of the code.

In short, running this notebook will train and evaluate *once* using *a single
set of parameters*, which includes *a single dataset and a single test
set*. It will also export TensorBoard summaries to a `logs` directory, for
example `~/Data/JIGSAWS/Suturing/logs`, making it easy to simultaneously
examine multiple runs with TensorBoard.

There are a two ways you can run the code:

1. Just run the notebook. This is convenient for immediate visualization and
experimentation, but inconvenient for batch runs. Every so often you will see
updates reflecting the training accuracy and raw edit distance as well as the
testing accuracy and raw edit distance. In addition, you'll see plots of
ground-truth labels vs. predicted labels for the test set as training
progresses.

2. `cd scripts && bash run_jigsaws_user_batch_sequential.sh`. This will run
train/eval sessions one after the other, populating many different Jupyter
notebooks with results and exporting TensorBoard summaries for each run.
