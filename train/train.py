import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.python.keras.saving.save import load_model
from wandb.keras import WandbCallback
import wandb
import kaggle
from timeit import default_timer as timer
from functions import load_labels, predict_on_testset, create_model, create_callbacks, evaluate_model


IMAGE_PATH_TRAIN = "../data/images/train-jpg/"
IMAGE_PATH_TEST = "../data/images/test-jpg/"


# * LOAD DATA

y_labels, UNIQUE_LABELS = load_labels()

train_dir = os.listdir(IMAGE_PATH_TRAIN)
test_dir = os.listdir(IMAGE_PATH_TEST)


# * INITIALIZE WANDB

run = wandb.init(project = "deforestation",
           reinit = True,
           name = "test_metric",
           config = {"cnn_layers": 4,
                     "filter_layout": "32-32-64-64",
                     "batch_norm": "2-4",
                     "max_pooling": "2-4",
                     "dense_layers": 1,
                     "dense_units": "512",
                     "full_data": True,
                     "data_size": None,
                     "epochs": 2,
                     "early_stop": True,
                     "batch_size": 32,
                     "activation": "elu",
                     "optimizer": "adam"})

config = wandb.config


# * CREATE MODEL

train_set, val_set = train_test_split(y_labels, test_size = 0.2)

m, train_generator, valid_generator = create_model(train_set, val_set, config)

early_stopping, checkpoint = create_callbacks()


# * RUN MODEL

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

history = m.fit_generator(generator = train_generator,
                steps_per_epoch = STEP_SIZE_TRAIN,
                validation_data = valid_generator,
                validation_steps = STEP_SIZE_VALID,
                epochs = config.epochs,
                class_weight = None,
                callbacks = [WandbCallback(), early_stopping, checkpoint]
                )
run.finish()


# * EVALUATE

# TODO state_full_binary... umbenennen

# m.save("models/6000-10epochs")
# from tensorflow.keras.models import load_model
# m = load_model("models/6000-10epochs")

y_train, y_train_pred, y_val, y_val_pred = evaluate_model(m, history, train_generator, valid_generator)


# * PREDICT

# Evaluate

submission = predict_on_testset(model = m, classes = train_generator.class_indices)

# submission.to_csv("./submissions/submission.csv")

!kaggle competitions submit -c planet-understanding-the-amazon-from-space -f submissions/submission.csv -m "First submit"

