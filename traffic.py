import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Display one image from each category
    # display_images_and_labels(images, labels)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Show architecture of the model
    model.summary()

    # Fit model on training data
    history = model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    plot_accuracy_loss(history)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print("Model saved to", filename, ".")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    # Initialize lists for images and labels
    images = []
    labels = []

    # Define folder path
    folder_path = os.path.join(".", data_dir)

    # Get list of labels
    categories = [f for f in os.listdir(folder_path)]

    # Iterate over list of categories
    for category in categories:

        # Define category path
        category_path = os.path.join(folder_path, category)

        # Get all directories
        dirs = os.listdir(category_path)

        for file in dirs:
            print('Loading category', category, ':', file, end="\r", flush=True)

            # Get file path
            file_path = os.path.join(folder_path, category, file)

            # Read image
            image = cv2.imread(file_path)
            # Resize image
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # Check if image size is 30x30
            if image.shape != (30, 30, 3):
                print('Category', category, 'File', file, ': size ERROR!')

            labels.append(category)
            images.append(image)

        print('Category', category, 'loaded!' + ' ' * 20)

    if len(images) == len(labels):
        print('\nSuccessfully loaded', len(labels), 'images from', len(categories), 'categories\n')
    else:
        print('ERROR! images and labels has different lengths!')

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Initialize the sequential model
    model = tf.keras.models.Sequential()

    # Add convolutional and polling layers - 32, 64, 128 total filters. 3,3 is the kernel
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Add flatten layer to change 3D output to 1D.
    model.add(tf.keras.layers.Flatten())

    # Add dropout layer for regularization.
    model.add(tf.keras.layers.Dropout(0.4))

    # Add dense or output layer.
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    # Compile the model suing default optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def display_images_and_labels(images, labels):
    # Display an image of each category.
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the last image for each label.
        image = images[labels.index(label)-1]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)), fontsize=5)
        i += 1
        _ = plt.imshow(image)
    plt.show()


def plot_accuracy_loss(history):
    # plotting graphs for accuracy
    plt.figure(0)
    plt.plot(history.history['accuracy'], 'g-')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.show()

    # plotting graphs for loss
    plt.figure(1)
    plt.plot(history.history['loss'], 'r-')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
