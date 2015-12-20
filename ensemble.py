#/usr/bin/python
import pandas as pd
import numpy as np

'''
ENSEMBLE = [
     ['kaggle_titanic_rf.csv', 1.],
     ['titanic_gbt_1217.csv', 1.],
     ['xgboost_845.csv', 1.],
     ['svm_rbf.csv', 1.]
]
'''
ENSEMBLE = [
     ['kaggle_titanic_rf.csv', .9],
     ['titanic_gbt_1217.csv', .6],
     ['xgboost_845.csv', .6],
     ['svm_rbf.csv', .4]
]

files = [f[0] for f in ENSEMBLE]
weight = pd.DataFrame(pd.Series([f[1] for f in ENSEMBLE], index = [f[0] for f in ENSEMBLE], name=0)) 
print(weight)

test_file = pd.read_csv('test.csv')
total_weight = sum([x[1] for x in ENSEMBLE])
ensemble_predictions = []
df_final = pd.DataFrame({'PassengerId': test_file['PassengerId'], })
                         
for f, _ in ENSEMBLE:
    df = pd.read_csv(f)   
    df_final[f] = df['Survived'] 

#for j in range(df_final.shape(0)):
#    for f in files:
#        df_final.loc[j, 'Survived'] = df_final.loc[j, 'Survived'] + weight
#print(df_final[files])
df_final['Survived'] = df_final[files].dot(weight) / total_weight
df_final.loc[df_final['Survived'] > .5, 'Survived'] = 1
df_final.loc[df_final['Survived'] <= .5, 'Survived'] = 0
df_final['Survived'] = df_final['Survived'].astype(int)
df_final = df_final.drop(files, axis=1)

#print(df_final)
df_final.to_csv('ensemble1_withsvm.csv', index=False)
