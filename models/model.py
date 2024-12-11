# models/model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    try:
        data = pd.read_csv('data/iris.csv')
        data.columns = ['feature', 'label']
        X = data[['feature']]
        y = data['label']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def train_model(X_train, y_train):
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions)
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        return None

if __name__ == "__main__":
    result = load_data()
    if result is not None:
        X_train, X_test, y_train, y_test = result
        model = train_model(X_train, y_train)
        if model is not None:
            accuracy = evaluate_model(model, X_test, y_test)
            if accuracy is not None:
                print(f"Model Accuracy: {accuracy}")