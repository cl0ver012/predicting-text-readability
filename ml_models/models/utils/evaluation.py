"""
Module for evaluation of machine learning models.
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score
from scipy.stats import spearmanr


def print_metrics(y_true, y_pred):
    
    print("================================================")
    print("Spearman's correlation coef: " + str(spearmanr(y_true, y_pred)[0]))
    print("================================================")
    
    print("-----------")
    print("R^2 = " + str(r2_score(y_true, y_pred)))
    print("R = " + str(np.sqrt(r2_score(y_true, y_pred))))
    print("-----------")
    
  
    
    
    