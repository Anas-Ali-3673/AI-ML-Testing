import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import SingleFeatureContribution

def test_data_validation():
    data = pd.read_csv('data/iris.csv')
    dataset = Dataset(data, label='species')
    check = SingleFeatureContribution()
    result = check.run(dataset)
    result.show()