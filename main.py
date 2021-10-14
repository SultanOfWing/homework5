# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('/Users/dmitry/Downloads/true_car_listings_prepeared.csv')

year = 2017
millage = 23405
make = 11
state = 6
city = 17


def price_function_by_three_params(x1, x2, x3, k1, k2, k3, b):
    return x1 * k1 + x2 * k2 + x3 * k3 + b


model = LinearRegression()
le = preprocessing.LabelEncoder()
# let use try to use uppercase to avoid data duplication
# df['Make'] = df['Make'].str.upper()
# df['City'] = df['City'].str.upper()
# df['State'] = df['State'].str.upper()
# df['Vin'] = df['Vin'].str.upper()
# Stranger thing: data uppercase slightly reduces accuracy

x_normalized = df[df['Price'] != 0]
x_drop_price = x_normalized.drop('Price', axis=1)

# let's assume that the biggest focus on price comes from year and mileage
# everyone who has chosen a car in the secondary market knows this
x_train_by_year_mileage = x_drop_price[['Mileage', 'Year']].dropna()
# so we drop all NaN rows
y_train_by_year_mileage = x_normalized.loc[x_train_by_year_mileage.index]['Price']
model.fit(x_train_by_year_mileage, y_train_by_year_mileage)
print(mean_squared_error(y_train_by_year_mileage, model.predict(x_train_by_year_mileage), squared=False))
# rmse = 12157
# let try to minimize rmse

x_train_by_year_mileage_make = x_drop_price[['Make', 'Year', 'Mileage']].dropna()
x_train_by_year_mileage_make['Make'] = le.fit_transform(x_train_by_year_mileage_make['Make'])
y_train_by_year_mileage_make = x_normalized.loc[x_train_by_year_mileage_make.index]['Price']
# print(model.predict(pd.DataFrame({'Year': [year], 'Mileage': [millage], 'Make': [make]})))
# 26078 dollars

model.fit(x_train_by_year_mileage_make, y_train_by_year_mileage_make)
print(mean_squared_error(y_train_by_year_mileage_make, model.predict(x_train_by_year_mileage_make), squared=False))
# rmse = 12113, so now it the most accurate prediction (if at all such an error can be considered accurate)


# lets use all features
x_train_by_all = x_drop_price[['City', 'Vin', 'Model', 'State', 'Year', 'Mileage', 'Make']].dropna()
x_train_by_all['Make'] = le.fit_transform(x_train_by_all['Make'])
x_train_by_all['Model'] = le.fit_transform(x_train_by_all['Model'])
x_train_by_all['City'] = le.fit_transform(x_train_by_all['City'])
x_train_by_all['State'] = le.fit_transform(x_train_by_all['State'])
x_train_by_all['Vin'] = le.fit_transform(x_train_by_all['Vin'])
y_train_by_all = x_normalized.loc[x_train_by_all.index]['Price']
model.fit(x_train_by_all, y_train_by_all)
print(mean_squared_error(y_train_by_all, model.predict(x_train_by_all), squared=False))
# mse = 12022: the most accurate model when using all features ¯\_(ツ)_/¯ ¯\_(ツ)_/¯ ¯\_(ツ)_/¯
# btw feature shuffle has no effect :(

# rmse = 12089: it looks better than default 'Year', 'Mileage' using, but insignificantly

# anyway rmse too big, so try find another approach, btw we know that vin should not influence the model
# now we try to improve model by categorical encoding
# also we add train_test
x_categorical = pd.get_dummies(data=x_drop_price[['Year', 'Mileage', 'Make']].dropna(), columns=['Make'])
y_categorical = x_normalized.loc[x_categorical.index]['Price']
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_categorical, y_categorical, test_size=0.8)
model.fit(x_train, y_train)
print(mean_squared_error(y_test, model.predict(x_test), squared=False))
# rmse = 9380 B.I.N.G.O!


# to consolidate the success let's try to add polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False)
x_poly = poly.fit_transform(x_train_by_year_mileage)
x_train_poly, x_test_poly, y_train_poly, y_test_poly = sklearn.model_selection.train_test_split(x_poly,
                                                                                                y_train_by_year_mileage,
                                                                                                test_size=0.8)
model.fit(x_train_poly, y_train_poly)
print(mean_squared_error(y_test_poly, model.predict(x_test_poly), squared=False))

print(x_normalized.describe())

# looks like we have 51 states!!!!
print(np.unique(x_normalized['State'].values))
# OK, its 50 states + District of Columbia

# let's count all invalid values
# NaN values are invalid
print(df.isnull().sum())
# 120943 rows of Mileage and 77122 of Model
# also vehicle cannot costs zero
print(df[df['Price'] == 0].size)
# 426464 vehicles are given for nothing

x_train_by_year = x_drop_price['Year'].dropna()
x_train_by_year_reshaped = x_train_by_year.values.reshape(-1, 1)
y_train_by_year = x_normalized.loc[x_train_by_year.index]['Price']
# plt.scatter(x_train_by_year, y_train_by_year)
model.fit(x_train_by_year_reshaped, y_train_by_year)
plt.plot(x_train_by_year, model.predict(x_train_by_year_reshaped))
print(mean_squared_error(y_train_by_year, model.predict(x_train_by_year_reshaped), squared=False))
# rmse = 12418
# plt.show()
