from keras.models import Sequential
from keras.layers import AveragePooling2D, BatchNormalization, Conv2D, Cropping2D, Dense, Dropout, Flatten, MaxPooling2D
from customlayers import Grayscale, Normalizer


def get(depths, gray=True, dropout=0.7, crop=(80, 40), rows=160, cols=320, channels=3, outputs=1,
        showSizes=False):
    # A sequential model... obviously
    model = Sequential()

    # Crop the top and bottom portion of the image
    add(model, "Cropping", showSizes, Cropping2D(cropping=(crop, (0, 0)), input_shape=(rows, cols, channels)))

    # If desired, convert to grayscale
    if gray:
        add(model, "Grayscale", showSizes, Grayscale())

    # Normalize the data by mapping from [0,255] to [-1,1]
    add(model, "Normalizer", showSizes, Normalizer())

    i = 0
    add(model, "Conv2D ", showSizes, Conv2D(depths[i], (7, 7), activation="relu", strides=(1, 2)))
    i += 1
    add(model, "MaxPool", showSizes, MaxPooling2D(pool_size=(1, 2)))

    add(model, "Conv2D ", showSizes, Conv2D(depths[i], (7, 7), activation="relu", strides=(1, 2)))
    i += 1
    add(model, "MaxPool", showSizes, MaxPooling2D(pool_size=(2, 2)))

    add(model, "Conv2D ", showSizes, Conv2D(depths[i], (5, 5), activation="relu", strides=(1, 1)))
    i += 1
    add(model, "MaxPool", showSizes, MaxPooling2D(pool_size=(2, 2)))

    add(model, "Flatten", showSizes, Flatten())
    add(model, "Dropout", showSizes, Dropout(dropout))
    add(model, "Dense  ", showSizes, Dense(depths[i], activation="relu"))
    i += 1
    add(model, "Dropout", showSizes, Dropout(dropout))
    add(model, "Dense  ", showSizes, Dense(depths[i], activation="relu"))
    i += 1
    add(model, "Dropout", showSizes, Dropout(dropout))
    add(model, "Dense  ", showSizes, Dense(outputs))
    return model


def add(model, name, showSizes, layer):
    model.add(layer)
    if showSizes:
        print('--------------')
        print(name)
        layer = list(model.layers)[-1]
        print(layer.input_shape, layer.output_shape)
