#/usr/bin/python
import pandas as pd
import numpy as np

ENSEMBLE = [
     ['kaggle_titanic_rf.csv', 1.],      
     ['titanic_gbt_1217.csv', 1.],
     ['xgboost_845.csv', 1.]
]

files = [f[0] for f in ENSEMBLE]
test_file = pd.read_csv('test.csv')
total_weight = sum([x[1] for x in ENSEMBLE])
ensemble_predictions = []
df_final = pd.DataFrame({'PassengerId': test_file['PassengerId'], })
                         
for f, weight in ENSEMBLE:
    df = pd.read_csv(f)   
    df_final[f] = df['Survived'] 

df_final['Survived'] = np.sum(df_final[files], axis=1) / total_weight
df_final.loc[df_final['Survived'] > .5, 'Survived'] = 1
df_final.loc[df_final['Survived'] <= .5, 'Survived'] = 0
df_final['Survived'] = df_final['Survived'].astype(int)
df_final = df_final.drop(files, axis=1)

print(df_final)
df_final.to_csv('ensemble.csv', index=False)
