"""
Functions for cross validation.
"""
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import numpy as np


def grid_search_cv_for_ensembles(model, max_depth_values, n_estimators_values, X, y, scoring_function, k=5, verbose=0):
    """
    Performs the grid search for n_estimators and max_depth hyperparameters. 
    For each value in the grid does the k-folded cross validation.
    """
    
    best_score = 0.0
    best_n_estimators = 1
    best_max_depth = 1
    
    for max_depth in max_depth_values: 
        for n_estimators in n_estimators_values:
            
            kf = KFold(n_splits=k, random_state=None, shuffle=True)

            fold = 1
            scores = []
            for train_index, test_index in kf.split(X):

                # get train and test set for the i-th fold
                X_train, X_test = X.loc[train_index], X.loc[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # train and predict
                model.set_hyperparams(max_depth, n_estimators)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                scores.append(scoring_function(y_test, y_pred))

                fold += 1
              
            score = np.mean(scores)
            
            if verbose > 0:
                print("score=" + str(score) + " | max_depth=" + str(max_depth) + " n_estimators=" + str(n_estimators))

            if score > best_score:
                best_score = score
                best_n_estimators = n_estimators
                best_max_depth = max_depth

    return best_max_depth, best_n_estimators


def find_best_C(model, c_values, X, y, scoring_function, k=5, verbose=0):
    
    best_score = 0.0
    best_c = 1.0
    
    for c in c_values: 
            
        kf = KFold(n_splits=k, random_state=None, shuffle=True)

        fold = 1
        scores = []
        for train_index, test_index in kf.split(X):

            # get train and test set for the i-th fold
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # train and predict
            model.set_hyperparams('linear', c)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            scores.append(scoring_function(y_test, y_pred))

            fold += 1

        score = np.mean(scores)

        if verbose > 0:
            print("score=" + str(score) + " | C=" + str(c))

        if score > best_score:
            best_score = score
            best_c = c

    return best_c
    