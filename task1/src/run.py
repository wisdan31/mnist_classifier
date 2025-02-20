import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Supress TF Info messages

import argparse
from models.classifier import MnistClassifier
from data_loader import load_mnist
from sklearn.metrics import accuracy_score, classification_report

def preprocess_data(X_train, X_test, model_type):
    """Reshapes data for model."""
    if model_type == "rf":
        return X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)
    return X_train, X_test  

def evaluate_model(y_true, y_pred):
    """Prints classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

def main():
    parser = argparse.ArgumentParser(description="MNIST Classifier")
    parser.add_argument(
        "--model",
        choices=["rf", "fnn", "cnn"],
        required=True)
    
    args = parser.parse_args()

    # Load dataset
    (X_train, y_train), (X_test, y_test) = load_mnist()
    X_train, X_test = preprocess_data(X_train, X_test, args.model)

    # Initialize and train model
    classifier = MnistClassifier(args.model)
    print(f"Training {args.model} model...")
    classifier.train(X_train, y_train)

    # Predict and evaluate
    print("Making predictions...")
    y_pred = classifier.predict(X_test)
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()
