
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from .base import MnistClassifierInterface

class ConvlutionalClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = Sequential([
            Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(32, activation="relu"),
            Dense(10, activation="softmax")
        ])

    def train(self, X_train, y_train, epochs=10, batch_size=128):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
