import numpy as np
import pandas as pd
import pandas as pd
from sklearn.linear_model import LassoCV

pd.options.display.max_rows = 1999
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_list = [df_train, df_test]

df_final = pd.concat(df_list)

df_train_age = df_final[~np.logical_or(df_final['Age'].isnull(), df_final['Fare'].isnull())].copy()
df_train_age.loc[df_train_age['Fare'].isnull(), 'Fare'] = df_train_age['Fare'].dropna().median()

df_train_age['Sex'] = df_train_age['Sex'].map({'female': 0, 'male':
    1}).astype(int)

df_train_age.head(10)

predictors = ['Sex', 'SibSp', 'Parch', 'Fare']
print(df_train_age[predictors])
print('==================')
print(df_train_age[df_train_age['Age'].isnull()])
model = LassoCV(cv=10).fit(df_train_age[predictors], df_train_age['Age'])

df_test = pd.read_csv("test.csv")
df_test.loc[df_test['Fare'].isnull(), 'Fare'] = df_test['Fare'].dropna().median()
df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male':
    1}).astype(int)
df_test['AgeFill'] = model.predict(df_test[predictors])

print(df_test[['Name', 'Sex', 'Age', 'AgeFill']])

