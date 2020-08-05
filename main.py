import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

sns.set()
plt.rc('text', usetex=False)

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

DOWNLOAD_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

housing = pd.read_csv(DOWNLOAD_URL)
# print housing heading
# print(housing.iloc[:5, :10])
# get no of entries, column names and column types
# print(housing.info())
# ocean_proximity is of type Text, and seems repetitive, let's see what different values has
# print(housing["ocean_proximity"].value_counts())
# let's look at the other fields, using the describe method for numerical fields
# print(housing.describe())

#
# housing.hist(bins=50, figsize=(15, 10))
# plt.show()

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# print(len(train_set))
# print(len(test_set))

# we need a representative data set, that is why we are dividing the data frame into income categories
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# print(housing)
# housing["income_cat"].hist()

# Now we are ready to do a stratified sampling based on the income category.

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Now we can remove the income_cat attibute
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

# Now let's look for correlations


# Another way to check for correlation between attributes is to use PAndas's scatter matrix

from pandas.plotting import scatter_matrix

# attributes=["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12,9))

# the most promising value for predicting the house value is the median income
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# the plot reveals horizontal ines around 500,000  , then around 450,000 and more
# we want to remove those districts to prevent the algorithm from learning to reproduce these data quirks
# we found correlations between atributes
# we also found attributes with tail-heavy distribution.

# before preparing the data, we may want to explore new combinations of data
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

## PREPARE THE DATASET FOR THE ALGORITHMS OF ML

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# data cleaning: drop replace nan values with the median
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# handling  text and categorical attributes
housing_cat = housing[["ocean_proximity"]]
print(housing_cat)

# convert categories from text to numbers
# from sklearn.preprocessing import OrdinalEncoder
# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded =  ordinal_encoder.fit_transform(housing_cat)

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(cat_encoder.categories_)

# create custom transformation to add the three new columns
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


#pipelining several transformations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "meadian")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler',StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

#so far, we have handled the categorical columns and the numerical columns separately.
#now it can be done aat once using ColumnTransformer

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer(
    ("num",num_pipeline, num_attribs),
    ("cat", OneHotEncoder, cat_attribs)
)

housing_prepared = full_pipeline.fit_transform(housing)