from models.random_forest import RandomForestMnist

class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == "rf":
            self.model = RandomForestMnist()
        else:
            raise ValueError("Unknown algorithm, choose from: 'rf'.")
        
    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)