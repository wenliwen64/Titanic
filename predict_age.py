import pandas as pd
import pandas as pd
from sklearn import linear_model

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_list = [df_train, df_test]

df_final = pd.concat(df_list)

print(df_final)
