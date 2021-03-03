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
from timeit import default_timer as timer
from functions import show_image, image_from_array, create_generator, build_model, load_labels, show_image_predictions, ypred_to_bool, f2_score, show_multilabel_confusion_matrix, StatefullMultiLabelFBeta, get_labels_from_generator, get_optimal_threshold, plot_precision_recall_allclasses


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

# * RUN MODEL

train_set, val_set = train_test_split(y_labels[:6000], test_size = 0.2)

train_generator = create_generator(train_set, IMAGE_PATH_TRAIN, batch_size = config.batch_size, shuffle = True, classes = UNIQUE_LABELS)

# TODO In Zukunft hier evtl. augmentation einfügen. Dann muss für den test_gen aber die Funktion verändert werden (keine augmentation)

valid_generator = create_generator(val_set, IMAGE_PATH_TRAIN, batch_size = config.batch_size, shuffle = True, classes = UNIQUE_LABELS)

m = build_model(config, len(UNIQUE_LABELS))

m.summary()


F2Score = StatefullMultiLabelFBeta(n_class = len(UNIQUE_LABELS), 
                                   beta = 2,
                                   threshold = 0.4)

m.compile(optimizer = config.optimizer, 
          loss = 'binary_crossentropy', 
          metrics = [F2Score, "AUC"])


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



# * EVALUATE

# m.save("models/6000-10epochs")
from tensorflow.keras.models import load_model
m = load_model("models/6000-10epochs")

m.evaluate(train_generator)
m.evaluate(valid_generator) # TODO: andere metrik?


# TODO geht das mit F2?
plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.legend()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


# TODO beste threshold?



# Get y_train
y_train = get_labels_from_generator(train_generator)

# Get y_pred
y_pred = m.predict_generator(train_generator,
                             steps = STEP_SIZE_TRAIN,
                             verbose = 1)

# Thresholding
best_threshold = get_optimal_threshold(y_train, y_pred) #! mit validation set machen!

y_pred_bool = ypred_to_bool(y_pred, 0.4) # TODO mit best threshold


# precision / recall

precision_score(y_train, y_pred_bool, average = "macro")
# macro: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# micro: Calculate metrics globally by counting the total true positives, false negatives and false positives.

precision, recall = plot_precision_recall_allclasses(y_train, y_pred, UNIQUE_LABELS)

print(classification_report(y_train, y_pred_bool, target_names = sorted(UNIQUE_LABELS)))

# F2 Score
f2_score(y_train, y_pred_bool)

# confusion matrix
show_multilabel_confusion_matrix(y_train, y_pred_bool, UNIQUE_LABELS)

# Show some predictions
show_image_predictions(train_generator, y_pred_bool, False)



# * PREDICT

# test_set = y_labels[6000:7000]

# # Evaluate

# test_generator = create_generator(test_set, IMAGE_PATH_TRAIN, batch_size = 1, shuffle = False, classes = UNIQUE_LABELS)

# m.evaluate(test_generator) # TODO: andere metrik?

# # Generate ypreds

# STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

# test_generator.reset()

# ypred = m.predict_generator(test_generator,
#                             steps = STEP_SIZE_TEST,
#                             verbose = 1)


# ypred_bool = ypred_to_bool(ypred, 0.4)

# results = pd.DataFrame(ypred_bool, columns = test_generator.class_indices, index = test_generator.filenames)

# # Generate ytrues
# # TODO ersetzen durch get_labels_from_generator()?
# ytrue_generator = create_generator(test_set, IMAGE_PATH_TRAIN, batch_size = len(test_set), shuffle = False, classes = UNIQUE_LABELS)
# img, ytrue = ytrue_generator.next()
# del img

# # Score

# f2_score(ytrue, ypred_bool)


# # confusion matrix
# show_multilabel_confusion_matrix(ytrue, ypred_bool, UNIQUE_LABELS)
   
# # Show some predictions
# show_image_predictions(test_generator, ypred_bool, False)

# # predision / recall
# from sklearn.metrics import precision_score
# precision_score(ytrue, ypred_bool, average = "macro")

# from sklearn.metrics import classification_report
# print(classification_report(ytrue, ypred_bool, target_names = sorted(UNIQUE_LABELS)))


