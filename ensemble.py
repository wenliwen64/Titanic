#/usr/bin/python
import pandas as pd
import numpy as np

def generate_rand(number=1, n_set=1):
    rand_list = [np.random.uniform(0, 1, number) for i in range(n_set)]
    return [k/sum(k) for k in rand_list]

'''
ENSEMBLE = [
     ['kaggle_titanic_rf.csv', 1.],
     ['titanic_gbt_1217.csv', 1.],
     ['xgboost_845.csv', 1.],
     ['svm_rbf.csv', 1.]
]
'''
FILES = ['kaggle_titanic_rf.csv', 'titanic_gbt_1217.csv', 'xgboost_845.csv', 'svm_rbf.csv', 'mlpclassifier.csv']
#WEIGHTS = generate_rand(4, 1)
WEIGHTS = [.9, .6, .7, .4, .4] 
ENSEMBLE = [(f, weight) for f, weight in zip(FILES, WEIGHTS)]

'''12-20
ENSEMBLE = [
     ['kaggle_titanic_rf.csv', .9],
     ['titanic_gbt_1217.csv', .6],
     ['xgboost_845.csv', .6],
     ['svm_rbf.csv', .4]
]
'''
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
#gprint(df_final[files])
df_final['Survived'] = df_final[files].dot(weight) / total_weight
df_final.loc[df_final['Survived'] > .5, 'Survived'] = 1
df_final.loc[df_final['Survived'] <= .5, 'Survived'] = 0
df_final['Survived'] = df_final['Survived'].astype(int)
df_final = df_final.drop(files, axis=1)

#print(df_final)
df_final.to_csv('ensemble1_withsvm_mlp.csv', index=False)
