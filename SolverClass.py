#!/usr/bin/python
import re
from scipy import stats
from operator import itemgetter
import numpy as np
import pandas as pd
from sklearn import linear_model, cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import cross_val_score
import xgboost as xgb

class SolverClass(object):
    def __init__(self, ori_ftrain, ori_ftest):
        self.ori_ftrain = ori_ftrain
        self.ori_ftest = ori_ftest
        self.train_raw_df = pd.read_csv(self.ori_ftrain)
        self.train_df = self.train_raw_df.copy()
        self.test_raw_df = pd.read_csv(self.ori_ftest)
        self.test_df = self.test_raw_df.copy()
        self.DATASET_DICT = {'train': self.train_df, 'test': self.test_df}
	self.age_mapping = {}
	self.PREDICTORS = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"]#, "Titles"]#, "FamilyId"]


    def learn_and_evaluate_train(self):
        print('Phase II. Model evaluation...\n')

        print('>> Training: engineering features...\n')
        self.feature_engineering_train()

        print('>> Training: preparing data for evaluation...\n')
        self.sklearn_dformat(dataset='train')

        print('>> Training: checking missing...\n')
        self.check_mv()

        print('>> xgb training data preparation...\n')
        self.xgb_dformat(dataset='train')

        print('>> MLP training data preparation...\n')
        self.mlp_dformat(dataset='train')

        print('>> Training: extratrees clf...\n')
	self.learn_and_predict_xrt(dataset='train')

	print('>> Training: evaluate sklearn-MLP...\n')
        self.learn_and_predict_mlp(dataset='train')

        print('>> Training: evaluate sklearn-svm...\n')
        self.learn_and_predict_svm(dataset='train')

        print('>> Training: evaluate sklearn-gbt...\n')
        self.learn_and_predict_gbt(dataset='train')

        print('>> Training: evaluate RandomForestClassifier...\n')
        self.learn_and_predict_rf(dataset='train')

        print('>> Training: xgboost...\n')
        self.learn_and_predict_xgb(dataset='train')

    def learn_and_predict_test(self):
        print('PhaseIII. Do machine learning on test dataset...\n')

        print('>> Test: engineering features...\n')
        self.feature_engineering_test()

        print('>> Sklearn data structure preparation...\n')
        self.sklearn_dformat(dataset='test')

        print('>> xgb test data preparation...\n')
        self.xgb_dformat(dataset='test')

	print('>> MLP test data preparation...\n')
        self.mlp_dformat(dataset='test')

        print('>> Predicting: extratrees clf...\n')
	self.learn_and_predict_xrt(dataset='test')

        print('>> Predicting: using sklearn-mlpclassifier...\n')
        self.learn_and_predict_mlp(dataset='test')

        print('>> Predicting: evaluate sklearn-svm...\n')
        self.learn_and_predict_svm(dataset='test')

        print('>> Predicting: using RandomForestClassifier on test dataset...\n')
        self.predictions_rf = self.learn_and_predict_rf(dataset='test').copy()
        #self.predicts_rf = self.learn_and_predict_rf(dataset='test').copy()
        #self.predicts_rf2 = self.learn_and_predict_rf(dataset='test2').copy()

        print(self.predictions_rf)
        print('>> Predicting: using xgboost...\n')
	self.learn_and_predict_xgb(dataset='test')

        print('>> Predicting: using sklearn-gbt&logistic-regression\n')
	self.learn_and_predict_gbt(dataset='test')
    
    def finish(self, model='rf'):
        PREDICTIONS_DICT = {'rf': self.predictions_rf}
	RESULTS_DICT = {'rf': 'kaggle_titanic_rf.csv'}
        submission = pd.DataFrame({
            'PassengerId': self.test_df['PassengerId'],
            'Survived': PREDICTIONS_DICT[model] 
        })
        submission.to_csv(RESULTS_DICT[model], index=False)

    def feature_engineering_train(self): #  apply on train dataset DONE!!!
        """
        Check and fill missing values if necessary, get to the final state of the raw data = self.train_data, self.test_data
        """

        # Make a copy of origianl data

        # Check missing values   
        self.check_mv(dataset='train')

        # Convert the string data to numeric data or something like that 
        self.dconversion(dataset='train')

        # Finally generate new features
        self.generate_new_features(dataset='train')

        # Fill the missing values based on different models
        self.fill_mv(dataset='train', method='category')

        predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Titles", "FamilyId"]
	#assert(predictors == self.PREDICTORS)
	predictors = self.PREDICTORS 
	self.train_df[predictors] = self.train_df[predictors].astype(float)
        


    def feature_engineering_test(self): #  apply on train dataset DONE!!!
        """
        Check and fill missing values if necessary, get to the final state of the raw data = self.train_data, self.test_data
        """

        # Make a copy of origianl data

        # Check missing values
        self.check_mv(dataset='test')

        # Convert the string data to numeric data or something like that 
        self.dconversion(dataset='test')

        # Fill the missing values based on different models
        self.generate_new_features(dataset='test')
        self.fill_mv(dataset='test', method='category')

        predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Titles", "FamilyId"]
	self.test_df[predictors] = self.test_df[predictors].astype(float)
        
    def dconversion(self, dataset='train'):  # DONE!!!
        ds = self.DATASET_DICT[dataset]
	ds.loc[ds['Sex']=='male', 'Sex'] = 0 
	ds.loc[ds['Sex']=='female', 'Sex'] = 1 
	ds['Sex'] = ds['Sex'].astype(int)
        ds['Embarked'] = ds['Embarked'].fillna('S')

        ds.loc[ds['Embarked']=='S', 'Embarked'] = 0
        ds.loc[ds['Embarked']=='C', 'Embarked'] = 1
        ds.loc[ds['Embarked']=='Q', 'Embarked'] = 2
	ds['Embarked'] = ds['Embarked'].astype(int);

	ds['Fare'] = ds['Fare'].fillna(ds['Fare'].median())  # only for test dataset in this case, or we can use kde 

    def check_mv(self, dataset='train'):  # DONE!!!
        print(self.DATASET_DICT[dataset].describe())

    def fill_mv(self, dataset='train', method='kde'):  # Train DONE!!!
        #self.learn_and_fill_age(dataset, method='median')
        #self.learn_and_fill_age(dataset, method='mean')
        self.learn_and_fill_age(dataset=dataset, method=method)
        #params0 = self.learn_age(df, dataset='median')
        #params1 = self.learn_age(df, dataset='linear_reg')
  
        #self.fill_age(df, option='median', params=params0)
        #self.fill_age(df, option='linear_reg', params=params1)

    def learn_and_fill_age(self, dataset, method='kde'):  # DONE!!!
        ds = self.DATASET_DICT[dataset]
	if method == 'median':
            ds['Age'] = ds['Age'].fillna(ds['Age'].median())
        elif method == 'mean':
	    ds['Age'] = ds['Age'].fillna(ds['Age'].mean())
        elif method == 'linear_reg':# we can do this as a side project. 
            train_age_df = ds[~ds['Age'].isnull()].copy()
	    train_nullage_df = ds[ds['Age'].isnull()].copy()
	    predictors = ['Pclass', 'Sex', 'Fare', 'Parch', 'SibSp']

            regr = linear_model.LinearRegression(normalize=True) 
            regr.fit(train_age_df[predictors], train_age_df['Age'])
            score = regr.score(train_age_df[predictors], train_age_df['Age'])

            for i in range(ds.shape[0]):
                if np.isnan(ds['Age'].iloc[i]):
                    ds['Age'].iloc[i] = regr.predict(ds[predictors].iloc[i]).copy()

            print('....Regression score on training data  = {}\n'.format(score))
	elif method == 'kde':
	    MISSING_AGE_DICT = {0: {1:27, 2:29,3:27}, 1: {1:27, 2:27, 3:27}}; # See notebook(Main.ipnb) for details
	    for gender in MISSING_AGE_DICT:
	        for pclass in MISSING_AGE_DICT[gender]:
		    age = MISSING_AGE_DICT[gender][pclass]
		    print('missing age to be filled is: {}'.format(age))
		    selector = np.logical_and(ds['Sex']==gender, ds['Pclass']==pclass) 
		    ds.loc[selector, 'Age'] = ds.loc[selector, 'Age'].fillna(age)
        elif method == 'category':
            selector = ds['Age'].isnull() 

	    if(dataset=='test'):
		print('==================================')
	        print(sum((ds['Age'].isnull()) & (ds['SibSp']>=2)))
	        print('==================================')

            #for k, v in self.age_mapping.items():
	         #ds.loc[(ds['Age'].isnull()) & (ds['SibSp']>=2) & (ds['Titles']==k), 'Age'] = v['sibsp_many'] if v['sibsp_many'] else v['sibsp_few'] #median
	         #ds.loc[(ds['Age'].isnull()) & (ds['SibSp']<2) & (ds['Titles']==k), 'Age'] = v['sibsp_few'] if v['sibsp_few'] else v['sibsp_many']#median
            ds.loc[(ds['Age'].isnull()) & (ds['SibSp']>=2), 'Age'] = 11 #median
	    ds.loc[(ds['Age'].isnull()) & (ds['SibSp']<2), 'Age'] = 28 #median

	    print(ds['Age'])
        else:
            print('....!!! Choose correct method')

    def generate_new_features(self, dataset='train'):  # DONE!!!
        ds = self.DATASET_DICT[dataset]
        self.family_id_mapping = {}
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] 
        ds['NameLength'] = ds['Name'].apply(lambda x: len(x))
        titles = ds['Name'].apply(self.get_title)
	print('titles:')
	print(titles[titles=='C'])
    
	title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Col': 7, 'Major': 8, 'Mlle': 9, 'Countess': 10, 'Ms': 11, 'Lady': 12, 'Jonkheer': 13, 'Don': 14, 'Mme': 15, 'Capt': 16, 'Sir': 17, 'Dona':18}

        for k, v in title_mapping.items():
            titles[titles == k] = v 
            ds['Titles'] = titles
	    print('Title {}: {}'.format(k, sum((ds['Titles']==v) & (ds['Age'].isnull()))))
	    print('Title {}: {}'.format(k, ds.loc[ds['Titles'] == v, 'Age'].median()))

        ds['Titles'] = ds['Titles'].astype(int)
	if dataset == 'train':
            for k, v in title_mapping.items():
                self.age_mapping[v] = {'sibsp_many':ds.loc[(ds['Titles']==v) & (ds['SibSp']>=2), 'Age'].median(), 'sibsp_few':ds.loc[(ds['Titles']==v) & (ds['SibSp']<2), 'Age'].median()} 

        print(self.age_mapping)
        family_ids = ds.apply(self.get_family_id, axis=1)
        family_ids[ds['FamilySize'] < 3] = -1
        ds['FamilyId'] = family_ids
 
    def sklearn_dformat(self, dataset='train'):  # DONE!!!
        return

    def xgb_dformat(self, dataset='train'):  # Done
        #predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Titles", "FamilyId"]
	predictors = self.PREDICTORS
        if dataset == 'train':
            ftrain = open('./titanic.train.dmatrix', 'w')

            for i in range(self.train_df.shape[0]):
                ftrain.write('{} '.format(self.train_df['Survived'].iloc[i])) # target variable
                for j in range(len(predictors)):
                    ftrain.write('{0}:{1} '.format(j, self.train_df[predictors[j]].iloc[i]))
                ftrain.write('\n')

            ftrain.close()
            self.DMatrix_train = xgb.DMatrix('./titanic.train.dmatrix')
	    #self.DMatrix_train.save_binary('titanic_train.buffer')
        elif dataset == 'test':
            ftest = open('./titanic.test.dmatrix', 'w')

            for i in range(self.test_df.shape[0]):
                ftest.write('-1 ')
                for j in range(len(predictors)):
                    ftest.write('{0}:{1} '.format(j, self.test_df[predictors[j]].iloc[i]))
		ftest.write('\n')

            ftest.close()
            self.DMatrix_test = xgb.DMatrix('./titanic.test.dmatrix')
	    #self.DMatrix_test.save_binary('titanic_test.buffer')

    def mlp_dformat(self, dataset='train'): 
	#predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "FamilyId"]
        predictors = self.PREDICTORS
        scaler = StandardScaler()

        if dataset == 'train':
            self.mlp_train_df = self.train_df.copy()
            scaler.fit(self.mlp_train_df[predictors]) 
            self.mlp_train_df[predictors] = scaler.transform(self.mlp_train_df[predictors])

	elif dataset == 'test':
            self.mlp_test_df = self.test_df.copy()
            scaler.fit(self.mlp_train_df[predictors]) 
            self.mlp_test_df[predictors] = scaler.transform(self.mlp_test_df[predictors])

    def learn_and_predict_xrt(self, dataset='train'):
	#predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "FamilyId"]
	predictors = self.PREDICTORS
     
        if dataset == 'train':
            #clf = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=1, max_features=2, random_state=0)
	    clf = ExtraTreesClassifier()
	    param_grid = {
	                 'n_estimators': [200, 250, 300, 350, 400, 450, 500], 
			 'max_depth': [None, 3, 4, 5, 6],
			 'min_samples_split': [1, 2, 3, 4, 5, 6],
			 'max_features': [1, 2, 3, 4]
	    }

	    #grid_search = GridSearchCV(clf, param_grid=param_grid, cv=3)
            #grid_search.fit(self.train_df[predictors], self.train_df['Survived'])
	    #report(grid_search.grid_scores_)

            #clf.fit(self.train_df[predictors], self.train_df['Survived'])
            #scores = cross_val_score(clf, self.train_df[predictors], self.train_df['Survived'], cv=3)
            #print('scores.mean = {}, scores.std = {}'.format(scores.mean(),scores.std()))
        elif dataset == 'test':
            clf = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=1, max_features=2, random_state=0)
            clf.fit(self.train_df[predictors], self.train_df['Survived'])
	    predictions = clf.predict(self.test_df[predictors])
	    predictions_proba = clf.predict_proba(self.test_df[predictors])
	    predictions_proba = [f[1] for f in predictions_proba]
	    print(predictions_proba)
            submission = pd.DataFrame({
	                 'PassengerId': self.test_df['PassengerId'],
			 'Survived': predictions.astype(int),
		    })
            submission.to_csv('xrt_1222.csv', index=False) 

            submission_proba = pd.DataFrame({
	                 'PassengerId': self.test_df['PassengerId'],
			 'Survived': predictions_proba,
		    })
            submission_proba.to_csv('xrt_1222_soft.csv', index=False) 

    def learn_and_predict_mlp(self, dataset='train'):
	#predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "FamilyId"]
	predictors = self.PREDICTORS
	
	if dataset == 'train':
	    param_dist = {
	        'algorithm': ['l-bfgs'],
		'learning_rate': ['constant'],
		'learning_rate_init': [0.05, 0.01, 0.005, 0.001], # what is this
		'hidden_layer_sizes': [(8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,)],
		'activation': ['logistic', 'tanh', 'relu'],
		'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.], 
		'verbose': [False],
		'max_iter': [200],
	    }

            clf = MLPClassifier()
	    '''
            random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=400, cv=3)
            random_search.fit(self.mlp_train_df[predictors], self.mlp_train_df['Survived'])

            report(random_search.grid_scores_)
	    '''  
        elif dataset == 'test':
	  
            clf = MLPClassifier(learning_rate='constant', learning_rate_init=0.01, verbose=False, hidden_layer_sizes=(8,), max_iter=200, alpha=0.01, activation='relu', random_state=1)
	    clf.fit(self.mlp_train_df[predictors], self.mlp_train_df['Survived'])
            predictions = clf.predict(self.mlp_test_df[predictors])
	    predictions_proba = clf.predict_proba(self.mlp_test_df[predictors])
	    predictions_proba = [f[1] for f in predictions_proba]

            submission = pd.DataFrame({
	                 'PassengerId': self.test_df['PassengerId'], 
			 'Survived': predictions,
		    }) 
            submission.to_csv('./mlpclassifier_1222.csv', index=False)

            submission_proba = pd.DataFrame({
	                 'PassengerId': self.test_df['PassengerId'], 
			 'Survived': predictions_proba,
		    }) 
            submission_proba.to_csv('./mlpclassifier_1222_soft.csv', index=False)
    '''
    ==============================Machine Learning==================================
    '''
    def learn_and_predict_gbt(self, dataset='train'):  # DONE!!!
	#predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "FamilyId"]
        predictors = self.PREDICTORS
        if dataset == 'train':
            algorithms = [
                [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
                [LogisticRegression(random_state=1), predictors]
		]

            kf = KFold(self.train_df.shape[0], n_folds=3, random_state=1)
            full_predictions = []
            for train, test in kf:
                cv_predictions = []
                for alg, predictors in algorithms:
                    train_predictors = self.train_df[predictors].iloc[train,:].astype(float)
                    train_targets = self.train_df['Survived'].iloc[train].astype(float)
            # Fit the algorithm using the cv training data.
                    alg.fit(train_predictors, train_targets)
            # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
                    test_predictions = alg.predict(self.train_df[predictors].iloc[test, :].astype(float))
                    cv_predictions.append(test_predictions)
            # The gradient boosting classifier generates better predictions, so we weight it higher.
                predictions = (cv_predictions[0] * 3 + cv_predictions[1]) / 4
                predictions[predictions > 0.5] = 1
                predictions[predictions <= 0.5] = 0
                predictions = predictions.astype(int)
                full_predictions.append(predictions)

            full_predictions = np.concatenate(full_predictions, axis=0) 
            auc = float(np.sum(full_predictions == self.train_df['Survived'])) / float(np.size(full_predictions))
            print('accuracy is equal to ====> {}'.format(auc))
            train_model = pd.DataFrame({
	                  'PassengerId': self.train_df['PassengerId'],
			  'Survived': full_predictions,
	    })
	    train_model.to_csv('./gbt_train.csv', index=False)

        if dataset == 'test':

            algorithms = [
                [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
                [LogisticRegression(random_state=1), predictors]
		]

            predictions_ensemble = []
            for alg, predictors in algorithms:
            # Fit the algorithm using the full training data.
                alg.fit(self.train_df[predictors], self.train_df["Survived"])
            # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
                predictions = alg.predict_proba(self.test_df[predictors].astype(float))[:,1]
                predictions_ensemble.append(predictions)

            final_predictions = (predictions_ensemble[0] * 3 + predictions_ensemble[1]) / 4
	    final_predictions_proba = final_predictions.copy()

            final_predictions[final_predictions > 0.5] = 1.
            final_predictions[final_predictions <= 0.5] = 0. 
            final_predictions = final_predictions.astype(int)
	    print('====> gbt predictions <====')
	    print(final_predictions)
	    submission = pd.DataFrame({
		    'PassengerId': self.test_df['PassengerId'],
		    'Survived': final_predictions
		    })
            submission.to_csv('titanic_gbt_1217.csv', index=False)

	    submission_proba = pd.DataFrame({
		    'PassengerId': self.test_df['PassengerId'],
		    'Survived': final_predictions_proba,
		    })
            submission_proba.to_csv('titanic_gbt_1217_soft.csv', index=False)


    def learn_and_predict_xgb(self, dataset='train'):
        '''
        Use xgboost to do work
        '''
	#predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Titles", "FamilyId"]
	predictors = self.PREDICTORS
        if dataset == 'train':
	    param_dist = {'max_depth': sp_randint(3, 10),
	                  'learning_rate': [0.01, 0.03, 0.1, 0.3, 1.0],
	                  'gamma': [0, 0.1, 0.2, 0.3],
			  'subsample': [.1, .2, .3, .4, 0.5],
			  'colsample_bytree': [.4, .5],
			  'objective': ['binary:logistic'],
			  'n_estimators': sp_randint(20, 150),
			  }

	    clf = xgb.XGBClassifier()
            #random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=500, cv=3)
	    #random_search.fit(self.train_df[predictors], self.train_df['Survived'])

	    #report(random_search.grid_scores_)
            params = {'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.5, 'n_estimators': 54, 'subsample': .3, 'gamma': 0, 'objective':'binary:logistic', 'eval_metric': 'auc'} #0.845, cv=3 
	    bst = xgb.train(params, self.DMatrix_train)
	    predictions = pd.Series(bst.predict(self.DMatrix_train))
	    predictions[predictions >= .5] = 1
	    predictions[predictions < .5] = 0
	    predictions = [int(x) for x in predictions.tolist()]

            train_model = pd.DataFrame({
	                  'PassengerId': self.train_df['PassengerId'],
			  'Survived': predictions,
	    })
	    train_model.to_csv('./xgb_train.csv', index=False)

        else: 
            params = {'max_depth': 6, 'learning_rate': 0.1, 'colsample_bytree': 0.5, 'n_estimators': 54, 'subsample': .3, 'gamma': 0, 'objective':'binary:logistic', 'eval_metric': 'auc'} #0.845, cv=3 
	    bst = xgb.train(params, self.DMatrix_train)
	    #clf = xgb.XGBClassifier(params)
	    #clf.fit(self.train_df[predictors], self.train_df['Survived'], verbose=True)
	    #print(self.test_df[predictors])
	    predictions = pd.Series(bst.predict(self.DMatrix_test))
	    predictions_proba = predictions.copy()
	    predictions[predictions >= .5] = 1
	    predictions[predictions < .5] = 0
	    predictions = [int(x) for x in predictions.tolist()]
	    print(predictions)
            submission = pd.DataFrame({
                    'PassengerId': self.test_df['PassengerId'],
		    'Survived': predictions 
		    })
            submission.to_csv("xgboost_845.csv", index=False)
	    
            submission_proba = pd.DataFrame({
                    'PassengerId': self.test_df['PassengerId'],
		    'Survived': predictions_proba,
		    })
            submission_proba.to_csv("xgboost_845_soft.csv", index=False)


    def learn_and_predict_svm(self, dataset='train'):
	#predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize",
	#              "Titles", "FamilyId"]
	predictors = self.PREDICTORS

        if dataset == 'train':
	    param_dist = {
	        'kernel': ['rbf'],
	        'C': [.01, .03, .1, .3, 1, 3, 10., 20],
	        'gamma': np.random.uniform(0, 1, 100),
	    }

            clf = SVC() 
	    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=100, cv=3)
            random_search.fit(self.train_df[predictors], self.train_df['Survived'])
            report(random_search.grid_scores_)
	    self.svm_params = top_params(random_search.grid_scores_)
	    print(self.svm_params)

            clf = SVC(kernel='rbf', C=10, gamma=0.00158)
            clf.fit(self.train_df[predictors], self.train_df['Survived'])
	    predictions = clf.predict(self.train_df[predictors])
            predictions[predictions > .5] = 1  
            predictions[predictions <= .5] = 0
            train_model = pd.DataFrame({
	                  'PassengerId': self.train_df['PassengerId'],
			  'Survived': predictions,
	    })
	    train_model.to_csv('./svm_train.csv', index=False)

        elif dataset  == 'test':
	    clf = SVC(kernel='rbf', C=10, gamma=0.00158)
            clf.fit(self.train_df[predictors], self.train_df['Survived'])
	    predictions = clf.predict(self.test_df[predictors])
	    predictions_proba = predictions.copy() # cannot predict probability. clf.predict_proba(self.test_df[predictors])
	    #predictions_proba = [f[1] for f in predictions_proba]
            predictions[predictions > .5] = 1  
            predictions[predictions <= .5] = 0
            submission = pd.DataFrame({
	                     'PassengerId': self.test_df['PassengerId'],
			     'Survived': predictions
		    })
            submission.to_csv('svm_rbf.csv', index=False)

	    submission_proba = pd.DataFrame({
	                     'PassengerId': self.test_df['PassengerId'],
			     'Survived': predictions_proba,
		    })
            submission_proba.to_csv('svm_rbf_soft.csv', index=False)


    def learn_and_predict_rf(self, dataset='train'):
	#predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize",
	#              "Titles", "FamilyId"]
	predictors = self.PREDICTORS
        if dataset == 'train':
	    param_dist = {'max_depth': [3, None],
	                  'max_features': sp_randint(1, 8),
			  'min_samples_split': sp_randint(1, 11),
			  'min_samples_leaf': sp_randint(1, 11),
			  'bootstrap': [True, False],
			  'criterion': ['gini', 'entropy'],
			  'n_estimators': sp_randint(10,200)}

            clf = RandomForestClassifier()

            #random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=500, cv=3)
	    #random_search.fit(self.train_df[predictors], self.train_df['Survived'])
	    #report(random_search.grid_scores_)
	    

            #alg = RandomForestClassifier(random_state=1, n_estimators=20, min_samples_split=4, min_samples_leaf=2)
            #scores = cross_validation.cross_val_score(alg, self.train_df[predictors], self.train_df['Survived'], cv=3)
            clf = RandomForestClassifier(random_state=1, n_estimators=109, min_samples_split=3, min_samples_leaf=6, criterion='gini', max_features=6, bootstrap=True)
            clf.fit(self.train_df[predictors], self.train_df['Survived'])
            predictions = clf.predict(self.train_df[predictors].astype(float))

            predictions[predictions > .5] = 1
            predictions[predictions <= .5] = 0
            full_predictions = predictions.astype(int).copy()

            train_model = pd.DataFrame({
		    'PassengerId': self.train_df['PassengerId'],
		    'Survived': full_predictions 
	    })
            train_model.to_csv('./rf_train.csv', index=False)

            """
            # Grid Search for the optimal parameters for random forest
            tuned_parameters = [{'n_estimators':[1, 10, 30, 100], 'min_samples_split':[2, 4, 8, 16], 'min_samples_leaf':[1, 2, 4, 6]}, ]   
            clf = GridSearchCV(RandomForestClassifier(random_state=1), tuned_parameters, cv=3)
            clf.fit(self.train_df[predictors], self.train_df['Survived'])
            print '=========Search for best parameters========'
            print ''.join(['Best fit parameters are ', '\n'], )
            print clf.best_params_
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
                """
            #print 'mean score of original one is : \n'
        
            #print np.mean(scores)
        elif dataset == 'test':
            #predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Titles", "FamilyId"]
	    predictors = self.PREDICTORS
            clf = RandomForestClassifier(random_state=1, n_estimators=109, min_samples_split=3, min_samples_leaf=6, criterion='gini', max_features=6, bootstrap=True)
            clf.fit(self.train_df[predictors], self.train_df['Survived'])
            predictions = clf.predict(self.test_df[predictors].astype(float))
	    predictions_proba = clf.predict_proba(self.test_df[predictors].astype(float))
	    predictions_proba = [f[1] for f in predictions_proba] 

            predictions[predictions > .5] = 1.
            predictions[predictions <= .5] = 0.
            self.test_predictions = predictions.astype(int).copy()

            submission_soft = pd.DataFrame({'PassengerId': self.test_df['PassengerId'], 'Survived': predictions_proba})
	    submission_soft.to_csv('rf_1222_soft.csv', index=False)

            return self.test_predictions
        
    def get_family_id(self, row):
        last_name = row['Name'].split(',')[0]
        family_id = '{0}{1}'.format(last_name, row['FamilySize'])
        if family_id not in self.family_id_mapping:
            if len(self.family_id_mapping) == 0:
                current_id = 1
            else:
                # Here the operator.itemgetter is used to tell the comparison algo to compare the value/second item in each dictionary entry, 
                current_id = (max(self.family_id_mapping.items(), key=itemgetter(1))[1]+1)
                self.family_id_mapping[family_id] = current_id

            self.family_id_mapping[family_id] = current_id

        return self.family_id_mapping[family_id]

    def get_title(self, name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	if title_search:
	    return title_search.group(1)
        else:
	    return ''
    
# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def top_params(grid_scores):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[0]
    return top_scores.parameters

if __name__ == '__main__':
    pd.options.display.max_rows = 999
    solver = SolverClass('train.csv', 'test.csv')
    solver.learn_and_evaluate_train()
    solver.learn_and_predict_test()
    solver.finish('rf')
