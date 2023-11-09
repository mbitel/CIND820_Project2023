import numpy as np
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
import os
os.chdir('C:/Users/bitel/PycharmProjects/CIND820_Project/CIND820_Project')
crime = pd.read_csv('cleanedcommunitiescrime.csv', sep=',')
##One feature, 'OtherperCap' was an object data type but was supposed to be a numeric value. Change feature to numeric value.
x = crime.drop('ViolentCrimesPerPop', axis = 1)
y = crime['ViolentCrimesPerPop']
#Data was split into training (0.7) and test (0.3) data. The following preprocessing procedures and modelling will occur on the training data.
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
pt = PowerTransformer(method='yeo-johnson')
x_train = pd.DataFrame(pt.fit_transform(x_train), columns=x_train.columns)
#The feature selection filter method spearman rank correlation coefficient was applied to the preprocessed training dataset.
#The top 10 features were selected.
correlation_matrix, _ = spearmanr(x_train, y_train)
selection = SelectKBest(score_func=lambda x_train, y_train: correlation_matrix[:-1, -1], k=10)
selection.fit(x_train, y_train)
selected_features = x_train.columns[selection.get_support()].tolist()
print(selected_features)
#Used a linear regression model to evaluate the selected features against test data.
#
model = SVR(kernel='linear')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - ((1 - r2) * (1993 - 1) / (1993 - 97 - 1))
print(mse, mae, r2, adj_r2)
#k-fold cross validation was applied to a linear regression model
kf = KFold(n_splits=10)
model = SVR(kernel='linear')
scores = []
for train_index, test_index in kf.split(x):
    x_train, x_test = x[selected_features].iloc[train_index], x[selected_features].iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    scores.append(score)
print(scores)