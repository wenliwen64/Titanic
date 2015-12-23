#/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def generate_rand(number=1, n_set=1):
    rand_list = [np.random.uniform(0, 1, number) for i in range(n_set)]
    rand_list = list(rand_list)
    return [k/sum(k) for k in rand_list]

'''
ENSEMBLE = [
     ['kaggle_titanic_rf.csv', 1.],
     ['titanic_gbt_1217.csv', 1.],
     ['xgboost_845.csv', 1.],
     ['svm_rbf.csv', 1.]
]
'''
FILES = ['kaggle_titanic_rf.csv', 'titanic_gbt_1217.csv', 'xgboost_845.csv', 'svm_rbf.csv', 'xrt_1222.csv'] #'mlpclassifier.csv']

TRAIN_FILES = ['rf_train.csv', 'gbt_train.csv', 'xgb_train.csv', 'svm_train.csv']
FEATURE_FILES = ['rf', 'gbt', 'xgb', 'svm']
#WEIGHTS = generate_rand(4, 1)
WEIGHTS = [.9, .7, .7, .5]#, .5] 
WEIGHTS_SET = generate_rand(4, 1000) 
#print('WEIGHTS_SET: {}'.format(WEIGHTS_SET))
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
train_file = pd.read_csv('train.csv')

#==============Democratic Voting================
df_final = pd.DataFrame({'PassengerId': test_file['PassengerId'], })
weights = [1, 1, 1, 1, 1]
for f in FILES:
    df = pd.read_csv(f)
    df_final[f] = df['Survived']
final_predictions = df_final[FILES].dot(weights)/sum(weights)
final_predictions[final_predictions > .5] = 1
final_predictions[final_predictions <= .5] = 0
submission = pd.DataFrame({'PassengerId': test_file['PassengerId'], 'Survived': final_predictions.astype(int)})
submission.to_csv('demo_voting_5models_1222.csv', index=False)

total_weight = sum([x[1] for x in ENSEMBLE])
ensemble_predictions = []
df_final = pd.DataFrame({'PassengerId': test_file['PassengerId'], })
                         
for f, _ in ENSEMBLE:
    df = pd.read_csv(f)   
    df_final[f] = df['Survived']

ensemble_train = pd.DataFrame({'PassengerId': train_file['PassengerId'], })
for f, feature in zip(TRAIN_FILES, FEATURE_FILES):
    df = pd.read_csv(f)
    ensemble_train[feature] = df['Survived'].astype(int)

ensemble_train.to_csv('ensemble_train.csv', index=False)

ensemble_test = pd.DataFrame({'PassengerId': test_file['PassengerId'], })
for f, feature in zip(FILES, FEATURE_FILES):
    df_test = pd.read_csv(f)
    ensemble_test[feature] = df_test['Survived'].astype(int)

print(ensemble_test.shape)
ensemble_train['Survived'] = train_file['Survived']
kf = KFold(train_file.shape[0], n_folds=3, random_state=1)
#clf = GradientBoostingClassifier(random_state=1, n_estimators=20, max_depth=3) 
'''
algorithms = [
    GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
    LogisticRegression(random_state=1)
    ]
'''
#===========================Testing different algos====================
algorithms = [SVC(kernel='rbf', C=10, gamma=.001)]
#clf = LogisticRegression(random_state=1)
final_predictions_train = []
for train, test in kf:
    cv_predictions = []
    for alg in algorithms:
        train_predictors = ensemble_train[FEATURE_FILES].iloc[train, :].astype(float)
        test_predictors = ensemble_train[FEATURE_FILES].iloc[test, :].astype(float)
        train_targets = ensemble_train['Survived'].iloc[train].astype(float)
        test_targets = ensemble_train['Survived'].iloc[test].astype(float)
        alg.fit(train_predictors, train_targets)
        test_predictions = alg.predict(test_predictors)
	cv_predictions.append(test_predictions)
    #predictions = (cv_predictions[0] * 3 + cv_predictions[1] * 1) / 4
    predictions = cv_predictions[0]
    predictions[predictions > .5] = 1
    predictions[predictions <= .5] = 0
    final_predictions_train.append(predictions.astype(int))

final_predictions_train = np.concatenate(final_predictions_train, axis=0)
print('finale_predictions_train = {}'.format(final_predictions_train))
auc = float(np.sum(final_predictions_train == train_file['Survived'])) / float(np.size(final_predictions_train))
print('accuracy is equal to ====> {}'.format(auc))

cv_predictions = []
for alg in algorithms:
    alg.fit(ensemble_train[FEATURE_FILES].astype(float), ensemble_train['Survived'])
    cv_predictions.append(alg.predict(ensemble_test[FEATURE_FILES].astype(float)))

#final_predictions_test = (cv_predictions[0] * 3 + cv_predictions[1] * 1) / 4
#final_predictions_test = clf.predict(ensemble_test[FEATURE_FILES])
final_predictions_test = cv_predictions[0]
final_predictions_test[final_predictions_test > .5] = 1
final_predictions_test[final_predictions_test <= .5] = 0 
submission = pd.DataFrame({
      'PassengerId': test_file['PassengerId'],
      'Survived': final_predictions_test,
})

submission.to_csv('svm_ensemble_12_21.csv', index=False)
#test_predictions = clf.predict()

#for j in range(df_final.shape(0)):
#    for f in files:
#        df_final.loc[j, 'Survived'] = df_final.loc[j, 'Survived'] + weight
#gprint(df_final[files])
'''2014-12-21
df_final['Survived'] = df_final[files].dot(weight) / total_weight
df_final.loc[df_final['Survived'] > .5, 'Survived'] = 1
df_final.loc[df_final['Survived'] <= .5, 'Survived'] = 0
df_final['Survived'] = df_final['Survived'].astype(int)
df_final = df_final.drop(files, axis=1)
'''
#print(df_final)
#df_final.to_csv('ensemble1_withsvm_mlp.csv', index=False)
#================================Testing different weights combo======================
auc_list = []
weight_max = [] 
auc_max = 0

for weight in WEIGHTS_SET:
    train_predictors = ensemble_train[FEATURE_FILES].astype(float)
    train_predictions = train_predictors.dot(weight)
    train_predictions[train_predictions > .5] = 1
    train_predictions[train_predictions <= .5] = 0
    auc = float(np.sum(train_predictions == train_file['Survived'])) / float(np.size(train_predictions))
    auc_list.append(auc)
    if auc > auc_max:
        auc_max = auc
        weight_max = weight

print('max_auc = {0}: {1}'.format(max(auc_list), list(weight_max)))

final_predictions_test = ensemble_test[FEATURE_FILES].astype(float).dot(weight_max)
final_predictions_test[final_predictions_test > 0.5] = 1
final_predictions_test[final_predictions_test <= 0.5] = 0
submission = pd.DataFrame({
             'PassengerId': test_file['PassengerId'],
	     'Survived': final_predictions_test.astype(int),
})
submission.to_csv('optimized_linear_weighting_ensemble_1222.csv', index=False)

# Soft Ensemble
SOFT_FILES = ['titanic_gbt_1217_soft.csv', 'xgboost_845_soft.csv', 'rf_1222_soft.csv', 'xrt_1222_soft.csv']
weights = [1, 1, 1, 1]
final_df = pd.DataFrame({'PassengerId': test_file['PassengerId'],})
for f in SOFT_FILES:
    df = pd.read_csv(f)
    final_df[f] = df['Survived']

final_df['Survived'] = final_df[SOFT_FILES].dot(weights) / sum(weights)
print('soft proba result:{}'.format(final_df['Survived']))
final_df.loc[final_df['Survived'] > .5, 'Survived'] = 1
final_df.loc[final_df['Survived'] <= .5, 'Survived'] = 0

print('soft result:{}'.format(final_df['Survived']))
submission = pd.DataFrame({'PassengerId': test_file['PassengerId'], 'Survived': final_df['Survived'].astype(int)})
submission.to_csv('soft_ensemble_except_mlpsvm.csv', index=False)
