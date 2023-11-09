import time
import tracemalloc
import timeit
start_time = time.time()
tracemalloc.start()
import os
import pandas as pd
os.chdir('C:/Users/bitel/PycharmProjects/CIND820_Project/CIND820_Project')
crime = pd.read_csv('cleanedcommunitiescrime.csv', sep=',')
##One feature, 'OtherperCap' was an object data type but was supposed to be a numeric value. Change feature to numeric value.
x = crime.drop('ViolentCrimesPerPop', axis = 1)
y = crime['ViolentCrimesPerPop']
#Data was split into training (0.7) and test (0.3) data. The following preprocessing procedures and modelling will occur on the training data.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=345)
##One feature, 'OtherperCap' was an object data type but was supposed to be a numeric value. Change feature to numeric value.
x_train['OtherPerCap'] = pd.to_numeric(x_train['OtherPerCap'], errors='coerce')
print(x_train.dtypes)
###Converted the missing/null values into column median values.
x_train = x_train.fillna(0)
y_train = y_train.fillna(0)
x_train = x_train.replace(0, x_train.median())
y_train = y_train.replace(0, y_train.median())
#Conducted Yeo and Johnson transformation on the dataset because numerous features were skewed distributions.
#In order to improve the predictor variables and possibly the model, this transformation was applied.
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
x_train = pd.DataFrame(pt.fit_transform(x_train), columns=x_train.columns)
#The feature selection embedded method Lasso was applied to the preprocessed training dataset.
#The top 10 features were selected.
from sklearn.feature_selection import SelectKBest
from scipy.stats import spearmanr
correlation_matrix, _ = spearmanr(x_train, y_train, axis=0, nan_policy='propagate', alternative='two-sided' )
selection = SelectKBest(score_func=lambda x_train, y_train: correlation_matrix[:-1, -1], k=10)
selection.fit(x_train, y_train)
newx_train = selection.transform(x_train)
selected_features = x_train.columns[selection.get_support()].tolist()
print(selected_features)
#k-fold cross validation was used to evaluate the Support Vector Regressor (SVR) model against 10 split samples.
#The average r2 scores was determined as a performance metric.
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
cv = KFold(n_splits=10, random_state=1, shuffle=True)
model = SVR()
score = cross_val_score(model, newx_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
mean_score = sum(score)/10
model.fit(newx_train, y_train)
print(mean_score)
newx_test = selection.transform(x_test)
y_pred = model.predict(newx_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - ((1 - r2) * (598 - 1) / (598 - 97 - 1))
print(mse, mae, r2, adj_r2)
snapshot = tracemalloc.take_snapshot()
end_time = time.time()
tracemalloc.stop()
memory = snapshot.statistics('lineno')
for stat in memory[:10]:
    print(stat)
print(end_time - start_time)
t = timeit.timeit(stmt='''import time
import tracemalloc
import timeit
start_time = time.time()
tracemalloc.start()
import os
import pandas as pd
os.chdir('C:/Users/bitel/PycharmProjects/CIND820_Project/CIND820_Project')
crime = pd.read_csv('cleanedcommunitiescrime.csv', sep=',')
##One feature, 'OtherperCap' was an object data type but was supposed to be a numeric value. Change feature to numeric value.
x = crime.drop('ViolentCrimesPerPop', axis = 1)
y = crime['ViolentCrimesPerPop']
#Data was split into training (0.7) and test (0.3) data. The following preprocessing procedures and modelling will occur on the training data.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=345)
##One feature, 'OtherperCap' was an object data type but was supposed to be a numeric value. Change feature to numeric value.
x_train['OtherPerCap'] = pd.to_numeric(x_train['OtherPerCap'], errors='coerce')
print(x_train.dtypes)
###Converted the missing/null values into column median values.
x_train = x_train.fillna(0)
y_train = y_train.fillna(0)
x_train = x_train.replace(0, x_train.median())
y_train = y_train.replace(0, y_train.median())
#Conducted Yeo and Johnson transformation on the dataset because numerous features were skewed distributions.
#In order to improve the predictor variables and possibly the model, this transformation was applied.
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
x_train = pd.DataFrame(pt.fit_transform(x_train), columns=x_train.columns)
#The feature selection embedded method Lasso was applied to the preprocessed training dataset.
#The top 10 features were selected.
from sklearn.feature_selection import SelectKBest
from scipy.stats import spearmanr
correlation_matrix, _ = spearmanr(x_train, y_train, axis=0, nan_policy='propagate', alternative='two-sided' )
selection = SelectKBest(score_func=lambda x_train, y_train: correlation_matrix[:-1, -1], k=10)
selection.fit(x_train, y_train)
newx_train = selection.transform(x_train)
selected_features = x_train.columns[selection.get_support()].tolist()
print(selected_features)
#k-fold cross validation was used to evaluate the Support Vector Regressor (SVR) model against 10 split samples.
#The average r2 scores was determined as a performance metric.
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
cv = KFold(n_splits=10, random_state=1, shuffle=True)
model = SVR()
score = cross_val_score(model, newx_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
mean_score = sum(score)/10
model.fit(newx_train, y_train)
print(mean_score)
newx_test = selection.transform(x_test)
y_pred = model.predict(newx_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - ((1 - r2) * (598 - 1) / (598 - 97 - 1))
print(mse, mae, r2, adj_r2)''', number=50)
print(t)


