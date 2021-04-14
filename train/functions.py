import collections
import pickle
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import (average_precision_score, classification_report,
                             fbeta_score, multilabel_confusion_matrix,
                             precision_recall_curve, precision_score)
from tensorflow.keras import backend as K
from tensorflow.keras.applications.nasnet import (NASNetMobile,
                                                  decode_predictions,
                                                  preprocess_input)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input,
                                     MaxPooling2D)
from tensorflow.keras.metrics import Metric
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  img_to_array, load_img)
from tqdm import tqdm

# * CONSTANTS

IMAGE_PATH_TRAIN = "../data/images/train-jpg/"
IMAGE_PATH_TEST = "../data/images/test-jpg/"


# * FUNCTIONS

def show_image(path):
    img = load_img(path)
    return img

def image_from_array(array):
    img = Image.fromarray(array.astype('uint8'), 'RGB')
    return img

def show_image_predictions(generator, ypred_bool, reset = False):
    if reset:
        generator.reset()
    for i in range(4):
        img, labels = generator.next()
        row = generator.batch_index
        labels = np.where(ypred_bool[row] == 1)[0]
        classes = list(generator.class_indices)
        labels = [classes[i] for i in labels]
        plt.subplot(2, 2, i+1)
        plt.imshow(img[0])
        plt.title(labels)
        plt.axis('off')

def load_labels():
    y_labels = pd.read_csv("../data/y_labels/train_v2.csv")
    y_labels["tags"] = y_labels["tags"].apply(lambda x:x.split(" "))
    y_labels["image_name"] = y_labels["image_name"].apply(lambda x: x + ".jpg")
    UNIQUE_LABELS = list(set([x for sublist in y_labels["tags"] for x in sublist]))
    
    return y_labels, UNIQUE_LABELS

def create_generator(df, directory, batch_size, shuffle, classes):
    
    datagen = ImageDataGenerator(rescale = 1./255.)
                                #    featurewise_center = True,
                                #    featurewise_std_normalization = True,
                                #    rotation_range = 20,
                                #    width_shift_range = 0.2,
                                #    horizontal_flip = True,
                                #    vertical_flip = True)

    generator = datagen.flow_from_dataframe(
        dataframe = df,
        directory = directory,
        x_col = "image_name",
        y_col = "tags",
        batch_size = batch_size,
        seed = 42,
        shuffle = shuffle,
        classes = classes,
        class_mode = "categorical",
        target_size = (256,256))
    
    return generator

def build_cnn(config, n_labels):
    
    m = Sequential([ 
        # Layer 1
        Conv2D(filters = 32, 
            kernel_size = (3, 3), 
            strides = (2, 2),
            padding = "same",
            input_shape = (256, 256, 3)),
        Activation(config.activation),
        # Layer 2
        Conv2D(filters = 32, 
            kernel_size = (3, 3), 
            strides = (2, 2),
            padding = "same"),
        MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"),
        Activation(config.activation),
        Dropout(0.2),
        BatchNormalization(),
        # Layer 3
        Conv2D(filters = 64, 
            kernel_size = (3, 3), 
            strides = (2, 2),
            padding = "same"),
        Activation(config.activation),
        Dropout(0.2),
        # Layer 4
        Conv2D(filters = 64, 
            kernel_size = (3, 3), 
            strides = (2, 2),
            padding = "same"),
        MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"),
        Activation(config.activation),
        Dropout(0.2),
        BatchNormalization(),
        Flatten(),
        # Output Layer
        Dense(units = 512, activation = "elu"),
        Dense(units = n_labels, activation = "sigmoid"),
        ])
    
    return m

def create_generator(df, directory, batch_size, shuffle, classes, transfer_learning):
    
    if transfer_learning == True:
        preprocessing_func = preprocess_input
        rescale_factor = 0
    else:
        preprocessing_func = None
        rescale_factor = 1./255.
    
    datagen = ImageDataGenerator(rescale = rescale_factor,
                                 preprocessing_function = preprocessing_func)
                                #  preprocessing_function = preprocessing_func)
                                #    featurewise_center = True,
                                #    featurewise_std_normalization = True,
                                #    rotation_range = 20,
                                #    width_shift_range = 0.2,
                                #    horizontal_flip = True,
                                #    vertical_flip = True)

    generator = datagen.flow_from_dataframe(
        dataframe = df,
        directory = directory,
        x_col = "image_name",
        y_col = "tags",
        batch_size = batch_size,
        seed = 42,
        shuffle = shuffle,
        classes = classes,
        class_mode = "categorical",
        target_size = (256,256) # tuple of integers (height, width), default: (256, 256). The dimensions to which all images found will be resized. 
        )
    
    return generator


def generate_generators(train_set, val_set, config, labels, transfer_learning):
        
    train_generator = create_generator(train_set, IMAGE_PATH_TRAIN, batch_size = config.batch_size, shuffle = True, classes = labels, transfer_learning = transfer_learning)

    # TODO In Zukunft hier evtl. augmentation einfügen. Dann muss für den val_gen aber die Funktion verändert werden (keine augmentation)

    valid_generator = create_generator(val_set, IMAGE_PATH_TRAIN, batch_size = 1, shuffle = False, classes = labels, transfer_learning = transfer_learning) # Changing batch size for evaluation doesn't really do anything, other than adjusting the memory footprint of the graph
    
    return train_generator, valid_generator


def create_model(config, labels, transfer_learning):
    
    if transfer_learning == False:
        
        m = build_cnn(config, len(labels))
        
    else:
        
        base_model = NASNetMobile(
            input_tensor = Input(shape = (256, 256, 3)),
            include_top = False,
            weights = "imagenet",
            pooling = None
        )
        
        base_model.trainable = False
        
        m = Sequential([
            base_model,
            Flatten(),
            Dense(50, activation = "relu"),
            Dense(len(labels), activation = "sigmoid")
        ])

    F2Score = MultiLabelFBeta(n_class = len(labels), 
                              beta = 2,
                              threshold = 0.4)

    m.compile(optimizer = config.optimizer, 
            loss = 'binary_crossentropy', 
            metrics = [F2Score, "AUC"])
    
    m.summary()
    
    return m



def create_callbacks(model_name, patience):

    early_stopping = EarlyStopping(monitor = "val_loss", 
                                   patience = patience
                                   )

    checkpoint = ModelCheckpoint(f'models/{model_name}.hdf5', 
                                 monitor = "val_loss",
                                 verbose = 1, 
                                 save_best_only = True, 
                                 save_weights_only = True)
    
    return early_stopping, checkpoint



def ypred_to_bool(ypred, threshold):
    ypred_bool = (ypred > threshold).astype(int)
    return ypred_bool

def f2_score(y_true, y_pred):
    """From https://www.kaggle.com/anokas/fixed-f2-score-in-python
    Try out:
    # true = np.array([[1, 0, 0, 0, 1]])
    # pred = np.array([[1, 0, 1, 1, 0]])
    # f2_score(true, pred)

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def show_multilabel_confusion_matrix(ytrue, ypred, labels):
    
    assert ypred[0].dtype == "int64", "Error: Labels have to be boolean"
    
    matrix = multilabel_confusion_matrix(ytrue, ypred) # row = true (0, 1), column = pred (0,1)

    plt.figure(figsize = (12,8))
    for i in range(len(labels)):
        plt.subplot(4, 5, i+1)
        plt.tight_layout()
        plt.imshow(matrix[i], cmap = "Greys")
        plt.title(labels[i])
        for (x, y), value in np.ndenumerate(matrix[i]):
            plt.text(x, y, f"{value:.0f}", va = "center", ha = "center", c = "red")
            
def plot_precision_recall_allclasses(y_train, y_pred, labels):
    
    precision = dict()
    recall = dict()
    thresholds = dict()
    average_precision = dict()
    for i in range(len(labels)):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(y_train[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_train[:, i], y_pred[:, i])

    plt.figure(figsize = (12, 8))
    for i in range(17):
        plt.subplot(4, 5, i+1)
        plt.tight_layout()
        plt.step(recall[i], precision[i], where = 'post')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(sorted(labels)[i])
        
    return precision, recall
            
class MultiLabelFBeta(Metric):
    """ From https://towardsdatascience.com/f-beta-score-in-keras-part-iii-28b1721fc442 """
    def __init__(self, n_class, beta, threshold, average = 'samples', epsilon = 1e-7, name = 'binary_fbeta', **kwargs):
        
        # initializing an object of the super class
        super(MultiLabelFBeta, self).__init__(name=name, **kwargs)
            
        # initializing atrributes
        self.tp = self.add_weight(name='tp', shape=(n_class,), initializer='zeros') # initializing true positives
        self.actual_positives = self.add_weight(name='ap', shape=(n_class,), initializer='zeros') 
        self.predicted_positives = self.add_weight(name='pp', shape=(n_class,), initializer='zeros')

        self.n_samples = self.add_weight(name='n_samples', initializer='zeros')
        self.sum_fb = self.add_weight(name='sum_fb', initializer='zeros')

        # initializing other atrributes that wouldn't be changed for every object of this class
        self.beta_squared = beta**2
        self.average = average
        self.n_class = n_class
        self.threshold = threshold
        self.epsilon = epsilon

    def update_state(self, ytrue, ypred, sample_weight=None):
        # casting ytrue float dtype
        ytrue = tf.cast(ytrue, tf.float32)
        
        # making ypred one hot encoded 
        ypred = tf.cast(tf.greater_equal(tf.cast(ypred, tf.float32), tf.constant(self.threshold)), tf.float32)
        
        if self.average == 'samples': # we are to keep track of only fbeta
            # calculate true positives, predicted positives and actual positives atrribute along the last axis
            tp = tf.reduce_sum(ytrue*ypred, axis=-1) 
            predicted_positives = tf.reduce_sum(ypred, axis=-1)
            actual_positives = tf.reduce_sum(ytrue, axis=-1)
            
            precision = tp/(predicted_positives+self.epsilon) # calculate the precision
            recall = tp/(actual_positives+self.epsilon) # calculate the recall
            
            # calculate the fbeta score
            fb = (1+self.beta_squared)*precision*recall / (self.beta_squared*precision + recall + self.epsilon)
            
            if sample_weight is not None: # if sample weight is available for stand alone usage
                self.fb = tf.reduce_sum(fb*sample_weight)
            else:
                n_rows = tf.reduce_sum(tf.shape(ytrue)*tf.constant([1, 0])) # getting the number of rows in ytrue
                self.n_samples.assign_add(tf.cast(n_rows, tf.float32)) # updating n_samples
                self.sum_fb.assign_add(tf.reduce_sum(fb)) # getting the running sum of fb
                self.fb = self.sum_fb / self.n_samples # getting the running mean of fb

        else:
            # keep track of true, predicted and actual positives because they are calculated along axis 0
            self.tp.assign_add(tf.reduce_sum(ytrue*ypred, axis=0)) 
            self.assign_add(predicted_positives = tf.reduce_sum(ypred, axis=0))
            self.actual_positives.assign_add(tf.reduce_sum(ytrue, axis=0)) 
            
    def result(self):
        if self.average != 'samples':
            precision = self.tp/(self.predicted_positives+self.epsilon) # calculate the precision
            recall = self.tp/(self.actual_positives+self.epsilon) # calculate the recall

            # calculate the fbeta score
            fb = (1+self.beta_squared)*precision*recall / (self.beta_squared*precision + recall + self.epsilon)
            if self.average == 'weighted':
                return tf.reduce_sum(fb*self.actual_positives / tf.reduce_sum(self.actual_positives))

            elif self.average == 'raw':
                return fb
            
            return tf.reduce_mean(fb) # then it is 'macro' averaging 
    
        return self.fb # then it is either 'samples' with or without sample weight

    def reset_states(self):
        self.tp.assign(tf.zeros(self.n_class)) # resets true positives to zero
        self.predicted_positives.assign(tf.zeros(self.n_class)) # resets predicted positives to zero
        self.actual_positives.assign(tf.zeros(self.n_class)) # resets actual positives to zero
        self.n_samples.assign(0)
        self.sum_fb.assign(0)
        
def get_labels_from_generator(generator):
    generator.reset()
    _, y_train = next(generator)
    for i in tqdm(range(int(generator.n/generator.batch_size)-1)):
        _, label = next(generator)
        y_train = np.append(y_train, label, axis = 0)
    
    return y_train


def get_optimal_threshold(true_label, prediction, iterations = 100):
    
    """From https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475"""
    
    
    def fbeta(true_label, prediction):
        return fbeta_score(true_label, prediction, beta = 2, average = 'samples')

    n_classes = true_label.shape[1]
    best_threshold = [0.2] * n_classes   
    for t in tqdm(range(n_classes)):
        best_fbeta = 0
        temp_threshold = [0.2] * n_classes
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshold[t] = temp_value
            temp_fbeta = fbeta(true_label, prediction > temp_threshold)
            if  temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshold[t] = temp_value
    return best_threshold


def format_for_submission(df):
    """Format a data frame in the way which is required for submission on Kaggle.

    Args:
        df (DataFrame): DataFrame with one column per class

    Returns:
        DataFrame: DataFrame with one column for all labels, separated by a space
    """
    preds = []
    for i in tqdm(range(df.shape[0])):
        subset = df.iloc[i]
        tags =  " ".join(subset[subset == 1].index.tolist())
        preds.append(tags)
    
    out = pd.DataFrame(index = df.index)
    out.index.name = "image_name"
    out["tags"] = preds
    return out


def predict_on_testset(model, classes, threshold):
    
    test_datagen = ImageDataGenerator(rescale = 1./255)

    # From https://stackoverflow.com/questions/57516673/how-to-perform-prediction-using-predict-generator-on-unlabeled-test-data-in-kera
    test_generator = test_datagen.flow_from_directory(
            '../data/images/',
            batch_size = 1,
            shuffle = False,
            classes = ["test-jpg", "test-jpg-additional"], # subfolders
            class_mode = None, # do not create labels
            target_size = (256,256))

    y_test_pred = model.predict(test_generator, verbose = 1)

    y_test_pred_bool = ypred_to_bool(y_test_pred, threshold)

    filenames = [re.findall(r"\/(.+).jpg", filename)[0] for filename in test_generator.filenames]

    results = pd.DataFrame(y_test_pred_bool, 
                           columns = classes, 
                           index = filenames)

    submission = format_for_submission(results)
        
    return submission


def evaluate_model(m, history, train_generator, valid_generator, labels):

    # FBeta / AUC
    print("Evaluating on training data...")
    train_loss, train_fbeta, train_auc = np.round(m.evaluate(train_generator), 2)
    
    print("Evaluating on validation data...")
    valid_loss, valid_fbeta, valid_auc = np.round(m.evaluate(valid_generator), 2)
    
    print(f"\nF2-Score Train: {train_fbeta}\nF2-Score Valid: {valid_fbeta}\nAUC Train: {train_auc}\nAUC Valid: {valid_auc}\n")

    # Plot development
    plt.plot(history.history['loss'], label = 'training_loss')
    plt.plot(history.history['val_loss'], label = 'validation_loss')
    plt.legend()

    plt.plot(history.history['binary_fbeta'], label = "training_fbeta")
    plt.plot(history.history['val_binary_fbeta'], label = "validation_fbeta")
    plt.legend()

    plt.plot(history.history['auc'], label = "training_auc")
    plt.plot(history.history['val_auc'], label = "validation_auc")
    plt.legend()
    
    print("Get y_train and y_train_pred...")

    # Get y_train
    y_train = get_labels_from_generator(train_generator)

    # Get y_pred
    y_train_pred = m.predict(train_generator,
                            verbose = 1)
    
    print("\nGet y_val and y_val_pred...")

    # Get y_val
    y_val = get_labels_from_generator(valid_generator)

    # Get y_pred
    y_val_pred = m.predict(valid_generator, verbose = 1)


    # Thresholding
    print("\nGet best thresholds...")
    best_threshold = get_optimal_threshold(y_val, y_val_pred)
    threshold_df = pd.DataFrame(best_threshold, index = train_generator.class_indices, columns = ["threshold"])

    print(f"\nBest thresholds:\n{threshold_df}\n")

    y_val_pred_bool = ypred_to_bool(y_val_pred, best_threshold)


    # precision / recall

    print(f"\nValidation precision score: \n{np.round(precision_score(y_val, y_val_pred_bool, average = 'macro'), 2)}\n")
    # macro: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # micro: Calculate metrics globally by counting the total true positives, false negatives and false positives.

    precision, recall = plot_precision_recall_allclasses(y_val, y_val_pred, labels) # TODO: Richtige labels

    print(classification_report(y_val, y_val_pred_bool, target_names = sorted(labels)))


    # F2 Score
    print(f"F2 Score: \n{f2_score(y_val, y_val_pred_bool)}\n")


    # confusion matrix
    show_multilabel_confusion_matrix(y_val, y_val_pred_bool, labels)


    # Show some predictions
    show_image_predictions(valid_generator, y_val_pred_bool, False)
    
    
    return y_train, y_train_pred, y_val, y_val_pred, best_threshold



def get_class_weights(y_labels, train_generator):
    # class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only).
    all_labels = sum(y_labels.tags.tolist(), [])
    counter = collections.Counter(all_labels)
    all_labels = pd.DataFrame(counter.values(), index = counter.keys(), columns = ["weight"])
    weights = 1 / all_labels * all_labels.max(axis = 0)
    
    class_indices = train_generator.class_indices
    class_indices = pd.DataFrame(class_indices.values(), index = class_indices.keys(), columns = ["class_index"])
    
    weight_dict = pd.merge(weights, class_indices, left_index = True, right_index = True).set_index("class_index").to_dict()
    
    pickle.dump(weight_dict["weight"], open("pickle/weight_dict.p", "wb"))
    
    return(weight_dict["weight"])
