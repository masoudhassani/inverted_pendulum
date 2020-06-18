from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, Activation, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam, RMSprop

class CNN:
    def __init__(self, input_shape, num_actions, learning_rate=0.001):
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(input_shape,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(num_actions, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
        print(self.model.summary())
      