from keras import layers
from keras import models

CNN = models.Sequential()

CNN.add(layers.Conv2D(8, (3, 3), activation='sigmoid', input_shape=(100, 100, 1)))
CNN.add(layers.MaxPooling2D((2, 2)))
CNN.add(layers.Conv2D(16, (3, 3), activation='relu'))
CNN.add(layers.MaxPooling2D((2, 2)))
CNN.add(layers.Conv2D(16, (3, 3), activation='relu'))
CNN.add(layers.MaxPooling2D((2, 2)))
CNN.add(layers.Conv2D(16, (3, 3), activation='relu'))

CNN.add(layers.Flatten())
CNN.add(layers.Dense(32, activation='relu'))
CNN.add(layers.Dense(64, activation='relu'))
CNN.add(layers.Dense(3, activation='softmax'))

CNN.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

CNN.summary()


