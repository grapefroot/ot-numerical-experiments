import numpy as np

def confusion_matrix(predictions, protected_attribute):
    n = predictions.shape[0]
    a = np.sum((predictions == 0) & (protected_attribute == 0)) / n
    b = np.sum((predictions == 1) & (protected_attribute == 0)) / n
    c = np.sum((predictions == 0) & (protected_attribute == 1)) / n
    d = np.sum((predictions == 1) & (protected_attribute == 1)) / n
    return a, b, c, d

def balanced_error_rate(y_true, y_pred, protected_attributes):
    pass

def likelihood_ratio(y_predicted, protected):
    #following https://arxiv.org/pdf/1412.3756.pdf
    a, b, c, d = confusion_matrix(y_predicted, protected)
    return d * (a + c) / c / (b + d)

def disparate_impact(y_true, protected):
    return 1. / likelihood_ratio(y_true, protected)
