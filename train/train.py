import pickle

from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback
import wandb
from functions import load_labels, predict_on_testset, create_model, create_callbacks, evaluate_model, generate_generators


# * LOAD DATA

y_labels, UNIQUE_LABELS = load_labels()

# weight_dict = get_class_weights(y_labels, train_generator)
weight_dict = pickle.load(open("pickle/weight_dict.p", "rb"))


# * INITIALIZE WANDB

run = wandb.init(project = "deforestation",
           reinit = True,
           name = "transfer learning #1",
           config = {"cnn_layers": None,
                     "filter_layout": None,
                     "batch_norm": None,
                     "max_pooling": None,
                     "dense_layers": None,
                     "dense_units": None,
                     "full_data": True,
                     "data_size": None,
                     "epochs": 100,
                     "patience": 3,
                     "finetuning": True,
                     "augmentation": False,
                     "class_weight": False,
                     "early_stop": True,
                     "transfer_learning": True,
                     "batch_size": 32,
                     "activation": "elu",
                     "optimizer": "adam"})

config = wandb.config


# * CREATE MODEL

train_set, val_set = train_test_split(y_labels[0:200], test_size = 0.2)

train_generator, valid_generator = generate_generators(train_set, val_set, config, UNIQUE_LABELS, transfer_learning = config.transfer_learning, augmentation = False)

m, base_model, F2Score = create_model(config, UNIQUE_LABELS, transfer_learning = config.transfer_learning)

early_stopping, checkpoint, reduce_lr = create_callbacks(model_name = wandb.run.name, patience = config.patience)

if config.class_weight = True:
  class_weight = weight_dict
else:
  class_weight = None


# * RUN MODEL

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

history = m.fit(train_generator,
                steps_per_epoch = STEP_SIZE_TRAIN,
                validation_data = valid_generator,
                validation_steps = STEP_SIZE_VALID,
                epochs = config.epochs,
                class_weight = class_weight,
                callbacks = [WandbCallback(), early_stopping] # checkpoint, reduce_lr
                )
run.finish()


# * FINETUNING

if config.finetuning == True:

  base_model.trainable = True
  m.summary()

  m.compile(optimizer = Adam(1e-5), # low learning rate 
            loss = 'binary_crossentropy', 
            metrics = [F2Score, "AUC"])

  epochs = 10

  history = m.fit(train_generator,
                  steps_per_epoch = STEP_SIZE_TRAIN,
                  validation_data = valid_generator,
                  validation_steps = STEP_SIZE_VALID,
                  epochs = epochs,
                  class_weight = class_weight
                  )


# * EVALUATE

y_train, y_train_pred, y_val, y_val_pred, best_threshold = evaluate_model(m, history, train_generator, valid_generator, UNIQUE_LABELS)


# * PREDICT

submission = predict_on_testset(model = m, classes = train_generator.class_indices, threshold = best_threshold, transfer_learning = config.transfer_learning)

submission.to_csv("./submissions/submission.csv")

!kaggle competitions submit -c planet-understanding-the-amazon-from-space -f submissions/submission.csv -m "Test submit"