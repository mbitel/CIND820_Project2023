import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Lasso, LassoCV
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
#The feature selection embedded method Lasso was applied to the preprocessed training dataset.
#The top 10 features were selected.
lassocv = LassoCV(cv=10)
lassocv.fit(x_train, y_train)
print(lassocv.alpha_)