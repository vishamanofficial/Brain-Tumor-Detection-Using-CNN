import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

#define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images within 20 degrees
    width_shift_range=0.2,  # randomly shift images horizontally within 20% of the width
    height_shift_range=0.2,  # randomly shift images vertically within 20% of the height
    horizontal_flip=True  # randomly flip images horizontally
)

# Define the morphological operation parameters
kernel_size = (3, 3)
iterations = 1

try:
    image_directory = "datasets/"

    no_tumor_images = os.listdir(image_directory+'no/')#no tumor images in this list
    yes_tumor_images = os.listdir(image_directory+'yes/')# tumor images in this list
    dataset = []
    label = []
    INPUT_SIZE = 64

    for i, image_name in enumerate(no_tumor_images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + 'no/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            # Apply morphological operation (e.g., erosion or dilation)
            image = cv2.erode(np.array(image), kernel_size, iterations=iterations)
            dataset.append(np.array(image))
            label.append(0)

    for i, image_name in enumerate(yes_tumor_images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + 'yes/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            # Apply morphological operation (e.g., erosion or dilation)
            image = cv2.erode(np.array(image), kernel_size, iterations=iterations)
            dataset.append(np.array(image))
            label.append(1)

    dataset = np.array(dataset)
    label = np.array(label)
    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

    x_train = normalize(x_train, axis=1)
    x_test = normalize(x_test, axis=1)
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    # Apply data augmentation to x_train
    datagen.fit(x_train)


    # Model Building
    model = Sequential()

#conv 2d(no of filters,kernel size,size of image)
    model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test))

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # Predict labels for test data
    # Predict labels for test data
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Convert categorical labels back to binary
    y_test_labels = np.argmax(y_test, axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred)

    # Create confusion matrix plot
    labels = ['No Tumor', 'Tumor']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()

    # Add title and axis labels
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Show the plot
    plt.show()
    # Model Training
    model.fit(datagen.flow(x_train, y_train, batch_size=16), verbose=1, epochs=10, validation_data=(x_test, y_test))

    # Train the model and store the history

    model.save('BrainTumor10EpochsCategorical.h5')

except Exception as e:
    print("An error occurred:", e)



