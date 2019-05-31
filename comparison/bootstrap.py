"""
Functions for performing the statistical comparison using boostrap.
"""
import numpy as np


def bootstrap_significance_testing(y_true, y_predA, y_predB, metric, n=int(1e5)):
    """
    Perform bootstrap significance testing.
    
    Null hypothesis is: A is no better than B on the population as a whole.
    Alternative hypothesis: A is better than B on the population as a whole.
    
    The return value is the p-value for this test.
    The bootstrap estimates the p-value though a combination of simulation and approximation.
    
    A small p-value indicates strong evidence against the null hypothesis.
    In other words, it gives evidence that A is better than B.
    
    Explaination in detail (section 2.2. Boostrap):
    Berg-Kirkpatrick, Taylor, David Burkett, and Dan Klein. "An empirical investigation of statistical significance in nlp."
    Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning.
    Association for Computational Linguistics, 2012.
    
    :param y_true: 
    :param y_predA: predictions of model A
    :param y_predB: predictions of model B
    :param metric: used metric, has to be a function of form f(y_true, y_pred)
    :param n: integer; the number of times to perform bootstrap resampling
    """
    v1 = metric(y_true, y_predA)
    v2 = metric(y_true, y_predB)
    d = 2 * (v1 - v2)

    s = 0

    l = len(y_true)
    for i in range(n):
        idx = np.random.choice(l, l)

        v1i = metric(y_true[idx], y_predA[idx])
        v2i = metric(y_true[idx], y_predB[idx])
        di = v1i - v2i

        if di > d:
            s += 1

    return s / n