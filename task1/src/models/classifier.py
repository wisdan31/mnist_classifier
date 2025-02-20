from models.random_forest import RandomForestMnist
from models.feed_forward import FeedForwardClassifier

class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == "rf":
            self.model = RandomForestMnist()
        elif algorithm == "fnn":
            self.model = FeedForwardClassifier()
        else:
            raise ValueError("Unknown algorithm, choose from: 'rf', 'fnn'")
        
    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        probabilities = self.model.predict(X_test)
        class_labels = probabilities.argmax(axis=1)     # Convert to class labels
        return class_labels