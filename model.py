import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn import metrics

from sklearn.metrics import mean_squared_error,mean_absolute_error

#df = pd.read_csv('car data.csv')
df = pd.read_csv('ford.csv')
df = df[['price', 'mileage', 'year','mpg', 'engineSize']]

X = df.drop(['price'],axis = 1)
y = df['price']

rf = RandomForestRegressor()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

#print(random_grid)


rf_random = RandomizedSearchCV(estimator = rf, 
param_distributions = random_grid,
scoring='neg_mean_squared_error', n_iter = 3, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)

y_pred = rf_random.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
import pickle
# open a file, where you ant to store the data
file = open('randomF.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)



