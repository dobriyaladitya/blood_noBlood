# coding: utf-8
"""
Implementing convolutional neural networks
Using data augmentation to decrease overfitting

"""

from keras.models import Sequential
from keras.layers import Convolution2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

#change the location accordingly
INPUT_DIR_TRAIN = "/Users/adityadobriyal/Downloads/Blood_noblood/training_set"
INPUT_DIR_TEST = "/Users/adityadobriyal/Downloads/Blood_noblood/test_set"
NUM_OF_EPOCH = 10

# Using data augmentation as the size of traiining data is small
# This method uses the Keras ImageDataGenerator utility to process the image data. It performs
# the below steps :
#    Reads all the picture files
#    Decodes the JPEG content to RGB grids of pixels as Keras APIs require numeric values to process
#    Converts these into floating-point tensors as Keras APIs reqiure ndArrays to process
#    Rescales the pixel values (between 0 and 255) to the [0, 1] interval
#      (as you know, neural networks prefer to deal with small input values).
def prepare_training_data():

    # shear_range is for randomly applying shearing transformations
    # zoom_range is for randomly zooming inside pictures
    # horizontal_flip is for randomly flipping half the images horizontally—relevant when there are no assumptions of horizontal asymmetry
    # fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift
    train_datagen_augmented = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.1,
                                       zoom_range = 0.2,
                                       horizontal_flip = True,
                                       fill_mode='nearest')

    train_img_data_gen = ImageDataGenerator(rescale = 1./255)
    test_img_data_gen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen_augmented.flow_from_directory(INPUT_DIR_TRAIN ,
                                                     target_size = (150, 150),
                                                     batch_size = 32,
                                                     class_mode = 'binary')

    testing_set = test_img_data_gen.flow_from_directory(INPUT_DIR_TRAIN ,
                                                     target_size = (150, 150),
                                                     batch_size = 32,
                                                     class_mode = 'binary')
    return training_set, testing_set


# Displayes the image along with the label. Can be used to verify the input data
def show_sample_data_with_labels(training_set, range_vl):
    import matplotlib.pyplot as plt
    x,y = training_set.next()
    for i in range(0,range_vl):
        image = x[i]
        label = y[i]
        print (label)
        plt.imshow(image)
        plt.show()


def define_the_model():
    model = Sequential()
    # input_shape (image_height, image_width,image_channels)=(150, 150, 3) where 150*150 is the image size and 3 is the RGB
    model.add(Convolution2D(filters = 32, kernel_size = (3,3), input_shape=(150,150,3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    return model


def plot_the_graph(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def my_image_identifier():
    # Prepares the training and test data
    # training_set,test_set = prepare_training_data()
    training_set, test_set = prepare_training_data()

    # Defines the model with layers
    model = define_the_model()

    # Selects the optimizer, validation function and the metrics
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

    print(model.summary())
    model.save('blooded_or_not_2')

    # Processes in the batches
    history = model.fit_generator(training_set,
                        steps_per_epoch=100,
                        nb_epoch = NUM_OF_EPOCH,
                        validation_data = test_set,
                        validation_steps = 50
                                  )
    x,y = next(test_set)
    predictions = model.predict(x)

    for i in range (0,len(predictions)):
        print(i,': ',y[i], '    ', predictions[i])

    plot_the_graph(history)


my_image_identifier()
