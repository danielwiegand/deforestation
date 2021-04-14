import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback
import wandb
from functions import load_labels, predict_on_testset, create_model, create_callbacks, evaluate_model, generate_generators, get_class_weights


# * LOAD DATA

y_labels, UNIQUE_LABELS = load_labels()

# weight_dict = get_class_weights(y_labels, train_generator)
weight_dict = pickle.load(open("pickle/weight_dict.p", "rb"))


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

train_generator, valid_generator = generate_generators(train_set, val_set, config, UNIQUE_LABELS, transfer_learning = True)

m = create_model(config, UNIQUE_LABELS, transfer_learning = True)

early_stopping, checkpoint = create_callbacks(model_name = wandb.run.name,
                                              patience = 10)


# * RUN MODEL

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

history = m.fit(train_generator,
                steps_per_epoch = STEP_SIZE_TRAIN,
                validation_data = valid_generator,
                validation_steps = STEP_SIZE_VALID,
                epochs = config.epochs,
                class_weight = weight_dict,
                callbacks = [WandbCallback(), early_stopping, checkpoint]
                )
run.finish()


# * EVALUATE

# m.save("models/6000-10epochs")
# from tensorflow.keras.models import load_model
# m = load_model("models/6000-10epochs")

y_train, y_train_pred, y_val, y_val_pred, best_threshold = evaluate_model(m, history, train_generator, valid_generator, UNIQUE_LABELS)


# * PREDICT

# Evaluate

submission = predict_on_testset(model = m, classes = train_generator.class_indices, threshold = best_threshold)

# submission.to_csv("./submissions/submission.csv")

!kaggle competitions submit -c planet-understanding-the-amazon-from-space -f submissions/submission.csv -m "First submit"