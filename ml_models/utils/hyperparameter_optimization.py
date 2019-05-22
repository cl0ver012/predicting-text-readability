"""
Functions for hyperparameter optimization.
"""
from evaluation import get_r

# cross-validation from sklearn
from sklearn.model_selection import KFold

# models
from sklearn.svm import SVC


def find_best_C_for_SVC(X_train, y_train, kernel, Cs):
    """
    Searches for the best hyperparameter C of the given list of Cs.
    The best one is defined as having a higher R score than others.
    """
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    index = 0

    bestC = 1.0
    bestR = 0.0
    for train_index, test_index in kf.split(X_train):
        X_train_, X_test_ = X_train.loc[train_index], X_train.loc[test_index]
        y_train_, y_test_ = y_train[train_index], y_train[test_index]

        svc = SVC(kernel=kernel, C=Cs[index])
        svc.fit(X_train_, y_train_)
        y_pred = svc.predict(X_test_)

        R = get_r(y_test_, y_pred)
        if R > bestR:
            bestR = R
            bestC = Cs[index]

        index += 1
    
    return bestC