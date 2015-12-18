#/usr/bin/python
import pandas as pd
import numpy as np

ENSEMBLE = [
     ['kaggle_titanic_rf.csv', 1.],      
     ['titanic_gbt_1217.csv', 1.],
     ['xgboost_845.csv', 1.]
]

test_file = pd.read_csv('test.csv')
total_weight = sum([x[1] for x in ENSEMBLE])
ensemble_predictions = []
df_final = pd.DataFrame({'PassengerId': test_file['PassengerId'], })
                         
for f, weight in ENSEMBLE:
    df = pd.read_csv(f)   
    df_final[f] = df['Survived'] 

print(ENSEMBLE[len(ENSEMBLE)-1][0])
#df_final['Survived'] = np.sum(df_final[ENSEMBLE[0][0]:ENSEMBLE[len(ENSEMBLE)-1][0]], axis=1)
print(df_final)
df_final.to_csv('ensemble.csv', index=False)
