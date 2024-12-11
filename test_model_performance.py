import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation

def test_model_performance():
    data = pd.read_csv('data/iris.csv')
    X = data.drop('species', axis=1)
    y = data['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    train_dataset = Dataset(pd.concat([X_train, y_train], axis=1), label='species')
    test_dataset = Dataset(pd.concat([X_test, y_test], axis=1), label='species')
    
    suite = model_evaluation()
    result = suite.run(train_dataset, test_dataset, model)
    result.show()