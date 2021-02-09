import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.python.keras.saving.save import load_model
from wandb.keras import WandbCallback
import wandb
from timeit import default_timer as timer
from functions import show_image, image_from_array, create_generator, build_model, load_labels, show_image_predictions, ypred_to_bool, f2_score


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
                     "epochs": 10,
                     "early_stop": True,
                     "batch_size": 32,
                     "activation": "elu",
                     "optimizer": "adam"})

config = wandb.config

# * RUN MODEL

train_set, val_set = train_test_split(y_labels[:6000], test_size = 0.2)

train_generator = create_generator(train_set, IMAGE_PATH_TRAIN, batch_size = config.batch_size, shuffle = True, classes = UNIQUE_LABELS)
# ? In Zukunft hier evtl. augmentation einfügen. Dann muss für den test_gen aber die Funktion verändert werden (keine augmentation)

valid_generator = create_generator(val_set, IMAGE_PATH_TRAIN, batch_size = config.batch_size, shuffle = True, classes = UNIQUE_LABELS)

m = build_model(config, len(UNIQUE_LABELS))

m.summary()

m.compile(optimizer = config.optimizer, 
          loss = 'binary_crossentropy', 
          metrics = ["accuracy"])

#! F2 auch per batch auswerten lassen

# * CALLBACKS

early_stopping = EarlyStopping(monitor = "val_loss", 
                               patience = 10
                               )
checkpoint = ModelCheckpoint(f'models/{wandb.run.name}.hdf5', 
                             monitor = "val_loss",
                             verbose = 1, 
                             save_best_only = True, 
                             save_weights_only = True)


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



# * PREDICT

from tensorflow.keras.models import load_model
m = load_model("models/6000-10epochs")

test_set = y_labels[6000:7000]

test_generator = create_generator(test_set, IMAGE_PATH_TRAIN, batch_size = 1, shuffle = False, classes = UNIQUE_LABELS)

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

test_generator.reset()

ypred = m.predict_generator(test_generator,
                             steps = STEP_SIZE_TEST,
                             verbose = 1)

m.evaluate(test_generator)

ypred_bool = ypred_to_bool(ypred, 0.4)

results = pd.DataFrame(ypred_bool, columns = test_generator.class_indices, index = test_generator.filenames)

ytrue_generator = create_generator(test_set, IMAGE_PATH_TRAIN, batch_size = len(test_set), shuffle = False, classes = UNIQUE_LABELS)
img, ytrue = ytrue_generator.next()
del img

f2_score(ytrue, ypred_bool)


#! anschauen
# siehe evaluation-klassifikation.md > multi-label klassifikation
https://stackoverflow.com/questions/50686217/keras-how-is-accuracy-calculated-for-multi-label-classification
https://stats.stackexchange.com/questions/12702/what-are-the-measure-for-accuracy-of-multilabel-data
https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff
https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html#sklearn.metrics.multilabel_confusion_matrix


    
show_image_predictions(test_generator, ypred_bool, False)


# * EVALUATE

m.save("models/6000-10epochs")

m.evaluate(train_generator)
m.evaluate(valid_generator)

plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.legend()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

