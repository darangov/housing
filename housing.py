import os
import tarfile
from six.moves import urllib
import pandas as pd

'''
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
'''

# Obtain data
'''
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
'''

# Call data function
'''
fetch_housing_data()
'''

# Load the data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head() # Top five rows in the dataframe

housing.info() # Description of the data

# Categories related to column ocean_proximity - its maybe a Categorical attribute
housing["ocean_proximity"].value_counts()

# Summary of the numerical attributes
housing.describe()

# Plot Histogram to get another feel of the type of data

# solo para usar en Jupyter: %matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()

# Creating a Test Set
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train data +", len(test_set), "test data")
# Esta no es la forma correcta de determinar train y test sets
# ya que al actualizar el dataset se estaria mezclando data
# de train y de test, es decir, no se respetaria su clasificacion
# inicial
