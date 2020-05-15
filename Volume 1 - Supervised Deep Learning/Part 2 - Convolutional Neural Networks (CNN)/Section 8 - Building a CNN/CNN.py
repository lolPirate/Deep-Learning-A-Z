from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model

# Building the CNN

# initializing the NN
classifier = Sequential()

# Convolution Layer (Step 1)
# 1. Number of filters = Number of feature maps (64)
# 2. Numbe of rows, columns used to determine filter dimension. (3,3)
# 3. Input Shape = (channel, row, column) channel = 1 for Bw and 3 for RGB. Tensorflow has reversed input shape (row, column, channel)
classifier.add(Conv2D(
    64, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Pooling Layer (Step 2)
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(
    32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatenning (Step 3)
classifier.add(Flatten())

# Fully Connected Layers (Step 4)
classifier.add(Dense(128, bias_initializer='uniform', activation='relu'))
classifier.add(Dense(1, bias_initializer='uniform', activation='sigmoid'))

# Compiling the network
classifier.compile(
    optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image Preprocessing

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

classifier.fit(
    train_generator,
    steps_per_epoch=800,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=200)

classifier.save('Basic_CNN.h5')