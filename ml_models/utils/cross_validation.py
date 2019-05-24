"""
Functions for cross validation.
"""
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import numpy as np


def avg_spearman_over_k_folds(X, y, model, k=5):
    """
    Calculates the mean value of Spearman correlation over k folds of k-folded cross-validation.
    """
    kf = KFold(n_splits=k, random_state=None, shuffle=True)
    
    fold = 1
    spearmans = []
    for train_index, test_index in kf.split(X):
        print("Fold " + str(fold) + "/" + str(k))
        
        # get train and test set for the i-th fold
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        spearmans.append(spearmanr(y_test, y_pred)[0])

        fold += 1
    
    return np.mean(spearmans)