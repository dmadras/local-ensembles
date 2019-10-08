# Detecting Extrapolation with Local Ensembles
David Madras, James Atwood, Alex D'Amour

See `local_ensembles_demo.ipynb` for a demonstration of the method on a toy model. You can run similar code with  `python local_ensembles_demo.py` to reproduce the experiments in the paper from "Visualizing Extrapolation Detection". (Fig 5.1a, b).

To run the local ensembles method on a pre-trained model, do the following:

1. Set up a virtual environment with the dependencies in `requirements.txt` (see below).
2. Choose a directory _DIR_ and save a checkpoint of the pre-trained model in _DIR_/ckpts. You may have to change the model loading code in `local_ensembles/load_model.py` to suit your model structure.
3. Save samples from the training, validation, in-distribution test, and OOD data sets _DIR_/tensors. Call the files `{train, valid, test, ood}_{x, y}.npy`.
4. Choose a number of Lanczos iterations _NUM_ITERS_ to run, and an interval _PROJECTION_STEP_ denoting which values of _m_ to calculate the extrapolation score for (if _PROJECTION_STEP_ = 10, the score will be calculated for _m_ = 1, 11, 21 ... _NUM_ITERS_)
4. Run `local_ensembles/run_local_ensembles_main.py` as follows (for a regression model):

```
python local_ensembles/run_local_ensembles_main.py --expdir=DIR --num_lanczos_iterations=NUM_ITERS \
    --projection_step=PROJECTION_STEP --model_type=MLP_regressor --run_baselines=False
```

For a binary classification model, run:

```
python local_ensembles/run_local_ensembles_main.py --expdir=DIR --num_lanczos_iterations=NUM_ITERS \
    --projection_step=PROJECTION_STEP --model_type=CNN_classifier --run_baselines=False
```

For a multiclass classification model with N classes, run:
```
python local_ensembles/run_local_ensembles_main.py --expdir=DIR --num_lanczos_iterations=NUM_ITERS \
    --projection_step=PROJECTION_STEP --model_type=CNN_classifier --run_baselines=False \
    --use_prediction_gradient=False --n_labels=N
```

Note: this code is in TF-Eager. The model must be a TF model; using this code is straightforward if the model was trained using TF-Eager. If it was trained in Graph mode you may have to load it as an Eager model.

## Setting up a virtual environment with pip

To create a virtual environment `le` for running this code, do the following (credit to Elliot Creager for these instructions):
```
mkdir ~/venv 
python3 -m venv ~/venv/le
```
where `python3` points to python 3.6.X. Then
```
source ~/venv/le/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
