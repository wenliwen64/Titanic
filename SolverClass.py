#!/usr/bin/python
import pandas as pd
import re
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.cross_validation import KFold
import numpy as np
import operator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.grid_search import GridSearchCV

class SolverClass(object):
    def __init__(self, ori_ftrain, ori_ftest):
        print 'happy'
        self.ori_ftrain = ori_ftrain
        self.ori_ftest = ori_ftest


    def learn_and_predict_train(self):
        print 'II. Model evaluation...\n'

        print '---->>>Engineer features...\n'
        self.feature_engineering_train()

        print '---->>>sklearn data structure preparation...\n'
        self.sklearn_dformat(option='train')

        self.check_mv()

        print '---->>>xgb data structure preparation...\n'
        #self.xgb_dformat(option='train')

        print '---->>>Do sklearn-GBT...\n'
        self.learn_and_predict_sklearn(option='train')

        print '---->>>Do RandomForestClassifier on train dataset...\n'
        self.learn_and_predict_rf(option='train')

        print '---->>>Do xgboost on train dataset...\n'
        #self.predicts_xgb_train = self.learn_and_predict_xgb(option='train')

    def learn_and_predict_test(self):
        print 'III. Do machine learning on test dataset...\n'
        print '---->>>Engineer features in test dataset...\n'
        self.feature_engineering_test()

        print '---->>>Sklearn data structure preparation...\n'
        self.sklearn_dformat(option='test')

        print '---->>>Predict using sklearn on test dataset...\n'
        self.predicts_sklearn_test = self.learn_and_predict_sklearn(option='test').copy()

        print '---->>>Predict using RandomForestClassifier on test dataset...\n'
        self.predicts_rf = self.learn_and_predict_rf(option='test').copy()
        self.predicts_rf2 = self.learn_and_predict_rf(option='test2').copy()

        print self.predicts_rf
        print self.predicts_rf2
        print self.test_predictions


        for a,b in zip(self.predicts_rf, self.predicts_rf2):
            if a != b:
                print 'Viola!'
    
    def finish(self):
        submission = pd.DataFrame({
            'PassengerId': self.test_df['PassengerId'],
            'Survived': self.test_predictions 
        })
        submission.to_csv('kaggle_gbt_1030.csv', index=False)

    def feature_engineering_train(self): #  apply on train dataset DONE!!!
        # check and fill missing values if necessary, get to the final state of the raw data = self.train_data, self.test_data
        self.train_raw_df = pd.read_csv(self.ori_ftrain)
        self.train_df = self.train_raw_df.copy()

        # Convert the string data to numeric data or something like that 
        self.dconversion(option='train')

        # Fill the missing values based on different models
        self.check_mv(option='train')
        self.fill_mv(option='train', method='median')
        self.generate_new_features(option='train')

    def feature_engineering_test(self): #  apply on train dataset DONE!!!
        # check and fill missing values if necessary, get to the final state of the raw data = self.train_data, self.test_data
        self.test_raw_df = pd.read_csv(self.ori_ftest)
        self.test_df = self.test_raw_df.copy()

        # Convert the string data to numeric data or something like that 
        self.dconversion(option='test')

        # Fill the missing values based on different models
        self.check_mv(option='test')
        self.fill_mv(option='test', method='median')
        self.generate_new_features(option='test')
        
    def dconversion(self, option='train'):  # DONE!!!
        if option == 'train':
            self.train_df.loc[self.train_df['Sex'] == 'male', 'Sex'] = 0 
            self.train_df.loc[self.train_df['Sex'] == 'female', 'Sex'] = 1 
            self.train_df['Embarked'] = self.train_df['Embarked'].fillna('S')
            self.train_df.loc[self.train_df['Embarked'] == 'S', 'Embarked'] = 0
            self.train_df.loc[self.train_df['Embarked'] == 'C', 'Embarked'] = 1
            self.train_df.loc[self.train_df['Embarked'] == 'Q', 'Embarked'] = 2
        else:
            self.test_df.loc[self.test_df['Sex'] == 'male', 'Sex'] = 0 
            self.test_df.loc[self.test_df['Sex'] == 'female', 'Sex'] = 1 
            self.test_df['Embarked'] = self.test_df['Embarked'].fillna('S')
            self.test_df.loc[self.test_df['Embarked'] == 'S', 'Embarked'] = 0
            self.test_df.loc[self.test_df['Embarked'] == 'C', 'Embarked'] = 1
            self.test_df.loc[self.test_df['Embarked'] == 'Q', 'Embarked'] = 2 
            self.test_df['Fare'] = self.test_df['Fare'].fillna(self.test_df['Fare'].median())   

    def check_mv(self, option='train'):  # DONE!!!
        if option == 'train':
            print self.train_df.describe()
        else:
            print self.test_df.describe()

    def fill_mv(self, option='train', method='median'):  # Train DONE!!!
        #self.learn_and_fill_age(option, method='median')
        #self.learn_and_fill_age(option, method='mean')
        self.learn_and_fill_age(option, method)
        #params0 = self.learn_age(df, option='median')
        #params1 = self.learn_age(df, option='linear_reg')
  
        #self.fill_age(df, option='median', params=params0)
        #self.fill_age(df, option='linear_reg', params=params1)

    def learn_and_fill_age(self, option, method='median'):  # DONE!!!
        if option == 'train':
            if method == 'median':
                self.train_df['Age'] = self.train_df['Age'].fillna(self.train_df['Age'].median())
            elif method == 'mean':
                self.train_df['Age'] = self.train_df['Age'].fillna(self.train_df['Age'].mean())
            elif method == 'linear_reg': 
                train_age_df = self.train_df[~self.train_df['Age'].isnull()].copy()
                train_nullage_df = self.train_df[self.train_df['Age'].isnull()].copy()
                predictors = ['Pclass', 'Sex', 'Fare', 'Parch', 'SibSp']

                regr = linear_model.LinearRegression(normalize=True) 
                regr.fit(train_age_df[predictors], train_age_df['Age'])
                score = regr.score(train_age_df[predictors], train_age_df['Age'])
                for i in range(self.train_df.shape[0]):
                    if np.isnan(self.train_df['Age'].iloc[i]):
                        self.train_df['Age'].iloc[i] = regr.predict(self.train_df[predictors].iloc[i]).copy()

                print ''.join(['Regression score on training data  = ', str(score)])
            else:
                print 'choose correct method'

                #self.train_df['Age'] = self.train_df['Age'].fillna(regr.predict(self.train_df[predictors]))
        elif option == 'test':
            if method == 'median':
                self.test_df['Age'] = self.test_df['Age'].fillna(self.test_df['Age'].median())
            elif method == 'mean':
                self.test_df['Age'] = self.test_df['Age'].fillna(self.test_df['Age'].mean())
            elif method == 'linear_reg':
                test_age_df = self.test_df[~self.test_df['Age'].isnull()].copy()
                predictors = ['Pclass', 'Sex', 'Fare', 'Parch', 'SibSp']
                regr = linear_model.LinearRegression(normalize=True)
                regr.fit(test_age_df[predictors], test_age_df['Age'])
                score = regr.score(test_age_df[predictors], test_age_df['Age'])
                for i in range(self.test_df.shape[0]):
                    if np.isnan(self.test_df['Age'].iloc[i]):
                        self.test_df['Age'].iloc[i] = regr.predict(self.test_df[predictors].iloc[i]).copy()
                print ''.join(['Regression score on training data  = ', str(score)])
            else:
                print 'choose correct method'

   
    def generate_new_features(self, option='train'):  # DONE!!!
        if option == 'train':
            self.family_id_mapping = {}
            self.train_df['FamilySize'] = self.train_df['SibSp'] + self.train_df['Parch'] 
            self.train_df['NameLength'] = self.train_df['Name'].apply(lambda x: len(x))
            titles = self.train_df['Name'].apply(get_title)
    
            title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Col': 7, 'Major': 8, 'Mlle': 9, 'Countess': 10, 'Ms': 11, 'Lady': 12, 'Jonkheer': 13, 'Don': 14, 'Mme': 15, 'Capt': 16, 'Sir': 17}

            for k, v in title_mapping.items():
                titles[titles == k] = v 
                self.train_df['Titles'] = titles

            family_ids = self.train_df.apply(self.get_family_id, axis=1)
            family_ids[self.train_df['FamilySize'] < 3] = -1
            self.train_df['FamilyId'] = family_ids
        else:
            self.family_id_mapping = {}
            self.test_df['FamilySize'] = self.test_df['SibSp'] + self.test_df['Parch'] 
            self.test_df['NameLength'] = self.test_df['Name'].apply(lambda x: len(x))
            titles = self.train_df['Name'].apply(get_title)
    
            title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Col': 7, 'Major': 8, 'Mlle': 9, 'Countess': 10, 'Ms': 11, 'Lady': 12, 'Jonkheer': 13, 'Don': 14, 'Mme': 15, 'Capt': 16, 'Sir': 17}

            for k, v in title_mapping.items():
                titles[titles == k] = v 
                self.test_df['Titles'] = titles

            family_ids = self.test_df.apply(self.get_family_id, axis=1)
            family_ids[self.test_df['FamilySize'] < 3] = -1
            self.test_df['FamilyId'] = family_ids
 
    def sklearn_dformat(self, option='train'):  # DONE!!!
        return

    def xgb_dformat(self, option='train'):  # DONE!!!

        if option == 'train':
            ftrain = open('./titanic.train.dmatrix', 'w')
            for i in range(self.train_df.shape[0]):
                ftrain.write(str(titanic['Survived'].iloc[i]))
                ftrain.write(' ')
                for j in range(len(predictors)):
                    newline = ''.join([str(j), ':', str(self.train_df[predictors[j]].iloc[i]), ' \n'])
                    ftrain.write(newline)
            self.DMatrix_train = xgb.DMatrix('./titanic.train.dmatrix')

        else:
            ftest = open('./titanic.test.dmatrix', 'w')
            for i in range(self.test_df.shape[0]):
                ftest.write(str(-1))
                ftest.write(' ')
                for j in range(len(predictors)):
                    newline = ''.join([str(j), ':', str(self.test_df[predictors[j]].iloc[i]), ' \n'])
                    ftest.write(newline)
            self.DMatrix_test = xgb.DMatrix('./titanic.test.dmatrix')

    '''
    ==============================Machine Learning==================================
    '''
    def learn_and_predict_sklearn(self, option='train'):  # DONE!!!
        if option == 'train':
            predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Titles", "FamilyId"]

            algorithms = [
                [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
                [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Titles", "Age", "Embarked"]]
            ]

            kf = KFold(self.train_df.shape[0], n_folds=3, random_state=1)
            full_predictions = []
            for train,test in kf:
                cv_predictions = []
                for alg, predictors in algorithms:
                    train_predictors = self.train_df[predictors].iloc[train,:].astype(float)
                    train_targets = self.train_df['Survived'].iloc[train].astype(float)
            # Fit the algorithm using the cv training data.
                    alg.fit(train_predictors, train_targets)
            # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
                    test_predictions = alg.predict(self.train_df[predictors].iloc[test,:].astype(float))
                    cv_predictions.append(test_predictions)
            # The gradient boosting classifier generates better predictions, so we weight it higher.
                predictions = (cv_predictions[0] * 3 + cv_predictions[1]) / 4
                predictions[predictions > 0.5] = 1
                predictions[predictions <= 0.5] = 0
                predictions = predictions.astype(int)
                full_predictions.append(predictions)

            full_predictions = np.concatenate(full_predictions, axis=0) 
            auc = float(np.sum(full_predictions == self.train_df['Survived'])) / float(np.size(full_predictions))
            print ''.join(['accuracy is equal to ====> ', str(auc)])

        if option == 'test':
            predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Titles", "FamilyId"]

            algorithms = [
                [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
                [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Titles", "Age", "Embarked"]]
            ]

            predictions_ensemble = []
            for alg, predictors in algorithms:
            # Fit the algorithm using the full training data.
                alg.fit(self.train_df[predictors], self.train_df["Survived"])
            # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
                predictions = alg.predict_proba(self.test_df[predictors].astype(float))[:,1]
                predictions_ensemble.append(predictions)

            final_predictions = (predictions_ensemble[0] * 3 + predictions_ensemble[1]) / 4
            final_predictions[final_predictions > 0.5] = 1.
            final_predictions[final_predictions <= 0.5] = 0. 
            final_predictions = final_predictions.astype(int)
            print final_predictions
            self.test_predictions = final_predictions.copy()
            return self.test_predictions


    def learn_and_predict_xgb(self, option='train'):
        '''
        Use xgboost to do work
         '''
        if option == 'train':
            self.xgb_params = {'max_depth':3, 'eta':0.5, 'silent':1, 'objective':'binary:logistic'}
            self.xgb_num_round = 20
            t = xgb.cv(self.xgb_params, self.DMatrix_train, self.xgb_num_round, nfold=4, metrics={'error'}, seed=0)
        else: 
            params = {'max_depth':3, 'eta':0.5, 'silent':1, 'objective':'binary:logistic'}
            num_round = 20

    def learn_and_predict_rf(self, option='train'):
        if option == 'train':
            predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Titles", "FamilyId"]
            alg = RandomForestClassifier(random_state=1, n_estimators=20, min_samples_split=4, min_samples_leaf=2)
            scores = cross_validation.cross_val_score(alg, self.train_df[predictors], self.train_df['Survived'], cv=3)
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
            print 'mean score of original one is : \n'
        
            print np.mean(scores)
        elif option == 'test':
            predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Titles", "FamilyId"]
            alg = RandomForestClassifier(random_state=1, n_estimators=20, min_samples_split=4, min_samples_leaf=2)
            alg.fit(self.train_df[predictors], self.train_df['Survived'])
            predictions = alg.predict(self.test_df[predictors].astype(float))

            predictions[predictions > .5] = 1.
            predictions[predictions <= .5] = 0.
            self.test_predictions = predictions.astype(int).copy()
            return self.test_predictions
        elif option == 'test2':
            predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Titles", "FamilyId"]
            alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=16, min_samples_leaf=1)
            alg.fit(self.train_df[predictors], self.train_df['Survived'])
            predictions = alg.predict(self.test_df[predictors].astype(float))

            predictions[predictions > .5] = 1.
            predictions[predictions <= .5] = 0.
            self.test_predictions = predictions.astype(int).copy()
            return self.test_predictions

    def get_family_id(self, row):
        last_name = row['Name'].split(',')[0]
        family_id = '{0}{1}'.format(last_name, row['FamilySize'])
        if family_id not in self.family_id_mapping:
            if len(self.family_id_mapping) == 0:
                current_id = 1
            else:
            # here the operator.itemgetter is used to tell the comparison algo to compare the value/second item in each dictionary entry, 
                current_id = (max(self.family_id_mapping.items(), key=operator.itemgetter(1))[1]+1)
                self.family_id_mapping[family_id] = current_id

            self.family_id_mapping[family_id] = current_id

        return self.family_id_mapping[family_id]

def get_title(name):
    title_search = re.search(' ([A-Za-z]*)\.', name)
    if title_search:
        return title_search.group(1)
    else:
        return ''
    

pd.options.display.max_rows = 999
print 'happy'
solver = SolverClass('train.csv', 'test.csv')
print 'happy'
solver.learn_and_predict_train()
solver.learn_and_predict_test()
solver.finish()
