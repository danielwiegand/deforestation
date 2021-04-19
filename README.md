# Understanding Amazon from Space

This is the code for a submission to the Kaggle challenge "Planet: Understanding the Amazon from Space" (https://www.kaggle.com/c/planet-understanding-the-amazon-from-space).

## The challenge

In this competition, satellite image chips with  have to be labeled with atmospheric conditions and various classes of land cover/land use.  Resulting algorithms will help to better understand  where, how, and why deforestation happens all over the world - and  ultimately how to respond.

## Data

The training data consists of 40,479 labeled images of 256x256 pixels. Test data consists of 61,191 images. The images have three color channels.

The training set is labelled for atmospheric conditions and land cover / land use. The following figure shows the available labels:

![Alt text](data/rainforest_chips.jpeg?raw=true "The image labels")

More information on the data: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data

## Implementation

The problem can be classified as **multi-label classification**, as each image can have several labels. The appropriate classification algorithm are convolutional neural networks which use binary cross-entropy as loss function. The number of output neurons corresponds to the number of classes which is 17 in this case.

The tech stack used comprised the Keras and Scikit-Learn libraries. As models, both a custom CNN and transfer learning with a pre-trained NASnet mobile were tried.

As the classes are highly imbalanced, `accuracy` is not a suitable success metric. Instead, the organisators of the challenge use the F2 score, which is a weighted balance of `precision`and `recall`.

## Results

The custom CNN with four convolutional layers with some additional Dropout and Batch normalization layers and a large Dense layer before the output layer achieved a F2 score of 0.87125. Using a pre-trained NASnet mobile CNN with an additional Dense (50 neurons) and Dropout layer before the output layer achieves a score of 0.90685. Adding class-weights to better adapt to underrepresented classes slightly degrades performance; the same is the case for additional image augmentation during training. The best result of **0.91585** as public score was achieved with a pre-trained NASnet mobile which was finetuned for additional ten epochs with a very low learning rate of 1e-5. This corresponds to the 374th place of the original composition (top 40%).

The model could be further improved by using the significantly more performant NASnet Large model, but this was not tried due to performance limitations. Additionally, it could be tried to combine the outcomes of several models (ensemble learning).

## How to use this repository

1. Clone this repository
2. Add a `data` folder to the cloned folder and add train and test images, as well as the y labels file available under https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
3. Execute `train.py` line by line (e.g. in Google Colab)

## License

MIT License
