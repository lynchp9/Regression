# Standard Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test_1)

#SVR (Needs Feature Scaling)
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test_1)
y_pred = sc_y.inverse_transform(y_pred)

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test_1)

#Getting Cost Functions and R squared for comparison
cost_test = []
for i in range(0,len(y_test_1)):
    a = y_test_1[i] - y_pred[i]
    cost_test.append(a)
cost_test = np.square(cost)
cost_test = np.sum(cost)
cost_test = cost_test / len(y_test_1)

cost_train = []
for i in range(0,len(y_train)):
    b = y_train[i] - regressor.predict(X_train)[i]
    cost_train.append(b)
cost_train = np.square(cost_train)
cost_train = np.sum(cost_train)
cost_train = cost_train / len(y_train)