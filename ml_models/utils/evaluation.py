"""
Module for evaluation of machine learning models.
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score
from scipy.stats import pearsonr


def get_r(y_true, y_pred):
    return np.sqrt(r2_score(y_true, y_pred))
    

def print_metrics(y_true, y_pred, classification=False):
    
    if classification:
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("==============")

        print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
        print("F1(micro)" + str(f1_score(y_true, y_pred, average='micro')))
        print("F1(macro)" + str(f1_score(y_true, y_pred, average='macro')))
        print("==============")
    
    print("R^2 = " + str(r2_score(y_true, y_pred)))
    print("R = " + str(np.sqrt(r2_score(y_true, y_pred))))
    print("Pearson's correlation coef: " + str(pearsonr(y_true, y_pred)[0]))
    
    