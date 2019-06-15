"""
Machine learning models utility functions.
"""
import numpy as np


def discretize(y_pred):
    """
    Converts the predicted results from a continuous variable to five readability levels.
    """
    
    for i in range(len(y_pred)):    
        if y_pred[i] < 0.5:
            y_pred[i] = 0.0
        elif y_pred[i] < 1.5:
            y_pred[i] = 1.0
        elif y_pred[i] < 2.5:
            y_pred[i] = 2.0
        elif y_pred[i] < 3.5:
            y_pred[i] = 3.0
        else:
            y_pred[i] = 4.0
            
    return y_pred