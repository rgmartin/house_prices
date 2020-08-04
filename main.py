import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int( len(data) * test_ratio )
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices [test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


sns.set()
plt.rc('text', usetex=False)

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)


DOWNLOAD_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

housing = pd.read_csv(DOWNLOAD_URL)
# print housing heading
print(housing.iloc[:5,:10])
# get no of entries, column names and column types
print(housing.info())
# ocean_proximity is of type Text, and seems repetitive, let's see what different values has
print(housing["ocean_proximity"].value_counts())
#let's look at the other fields, using the describe method for numerical fields
print(housing.describe())

#
# housing.hist(bins=50, figsize=(15,10))
# plt.show()

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set))
print(len(test_set))

