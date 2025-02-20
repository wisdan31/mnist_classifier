from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from .base import MnistClassifierInterface

class FeedForwardClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = Sequential([
            Flatten(input_shape=(28,28)),
            Dense(128, activation="relu"),
            Dense(32, activation="relu"),
            Dense(10, activation="softmax")
        ])

    def train(self, X_train, y_train, epochs=50, batch_size=128):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
