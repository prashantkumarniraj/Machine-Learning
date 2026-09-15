import pandas as pd

house = pd.read_csv(
    'https://github.com/prashantkumarniraj/Machine-Learning/raw/refs/heads/main/Boston%20(1)%20(1).csv'
)

house.head()
house.head()

house.info()

house.describe()

house.columns

y = house['MEDV']

x = house.drop(['MEDV'], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=2529
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
model.intercept_
model.coef_
model.intercept_
model.coef_
y_pred = model.predict(X_test)
y_pred
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error
)
mean_absolute_error(y_test, y_pred)
mean_absolute_percentage_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)
