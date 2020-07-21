#import csv table and analyze raw data
from csv_import import data
from csv_import import descriptive

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

# This function splits the data on train and test subsets
def split_data(df, target='loan_status', size=0.4, seed=121):
    X = df.drop(target, axis = 1)
    y = df[[target]]
    result = train_test_split(X, y, test_size=size, random_state=seed)
    return (result)



# Display information about the dataframe and missing values
print('CSV file with {} rows imported'.format(data.shape[0]))
print('{} data instances for {} entries is missing'.
      format(data.isnull().sum().sum(),
             data[data.isnull().any(axis=1)].shape[0]))
print('')
print('-----------------')
print('')

# Remove obvious outliers: max_age=100, max_emp_length=60
max_age=100
max_emp_length=60
indices=data.query('person_age>{} or person_emp_length>{}'.format(max_age, max_emp_length)).index.tolist()
data_clean0 = data.drop(indices)

# Remove rows with missing interest rate data
indices=data_clean0[data_clean0['loan_int_rate'].isnull()].index
data_clean = data_clean0.drop(indices)

# Replace missing data on employment length with median values
data_clean['person_emp_length']\
.fillna((data_clean['person_emp_length'].median()),inplace=True)


