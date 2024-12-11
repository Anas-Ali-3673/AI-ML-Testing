import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    # Load data from a CSV file
    data = pd.read_csv('C:/Users/ALI COMPUTERS/Desktop/my_python_project/data/iris.csv')
    X = data[['feature']]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Dummy implementation for example purposes
    model = "trained_model"
    return model

def evaluate_model(model, X_test, y_test):
    # Dummy implementation for example purposes
    accuracy = 0.9
    return accuracy

def main():
    print("hey i am Anas and i am a python developer")

    # Call the model in the main to train it and evaluate it
    X_train, X_test, y_train, y_test = load_data('data.csv')
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

if __name__ == "__main__":
    main()