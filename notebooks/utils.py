import numpy as np
from scipy import stats

def confusion_matrix(predictions, protected_attribute):
    n = predictions.shape[0]
    a = np.sum((predictions == 0) & (protected_attribute == 0)) / n
    b = np.sum((predictions == 0) & (protected_attribute == 1)) / n
    c = np.sum((predictions == 1) & (protected_attribute == 0)) / n
    d = np.sum((predictions == 1) & (protected_attribute == 1)) / n
    return a, b, c, d

def balanced_error_rate(y_true, y_pred, protected_attributes):
    pass

def likelihood_ratio(y_predicted, protected):
    #following https://arxiv.org/pdf/1412.3756.pdf
    a, b, c, d = confusion_matrix(y_predicted, protected)
    return d * (a + c) / c / (b + d)

def disparate_impact(y_predicted, protected):
    return 1. / likelihood_ratio(y_predicted, protected)

def estimate_Tn(target_label:np.array, protected_variable:np.array, transform_func):
    """
    target_label:np.array[n] - may be either a classifier output or a given target label a.k.a g(X) in the paper
    protected_variable:np.array[n] - a.k.a S in the paper
    transform_func: protected_variable.dtype -> {0, 1} - transform the protected variable to {0, 1}
    
    Returns: 
    T_n: float - the value of T_n
    """
    if transform_func is None:
        transformed_protected = protected_variable
    else:
        transformed_protected = transform_func(protected_variable)
        
    numerator = sum((target_label == 1) & (transformed_protected == 0)) * sum(transformed_protected == 1)
    denominator = sum((target_label == 1) & (transformed_protected == 1)) * sum(transformed_protected == 0)
    
    T_n = numerator / denominator
    
    pi_0 = np.mean(transformed_protected == 0)
    pi_1 = np.mean(transformed_protected == 1)
    p_0 = np.mean((target_label == 1) & (transformed_protected == 0))
    p_1 = np.mean((target_label == 1) & (transformed_protected == 1))
    
    phi_grad = np.array([
        pi_1 / (pi_0 * p_1),
        -p_0 * pi_1 / (p_1**2 * pi_0),
        -p_0 * pi_1 / (p_1 * pi_0**2),
        p_0 / (p_1 * pi_0)
    ])
    sigma_mat = np.array([
        [p_0 * (1 - p_0), 0, 0, 0],
        [-p_0*p_1, p_1*(1 - p_1), 0, 0],
        [pi_1*p_0, -pi_0*p_1, pi_0*pi_1, 0],
        [-pi_1*p_0, pi_0*p_1, -pi_0 * pi_1, pi_0 * pi_1]])
    
    sigma_scalar = np.sqrt(phi_grad @ sigma_mat @ phi_grad)

    #print(sigma_mat)
    #print(phi_grad)
    return T_n, sigma_scalar

def construct_confidence_interval(target_label, protected_variable, transform_func=None, level=0.9):
    """
    Constructs the confidence interval for DI at level "level"
    
    Returns:
    T_n: float - the value of T_n
    lower_value:float - lower interval value
    upper_value:float - upper interval value
    """
    T_n, sigma_scalar = estimate_Tn(target_label, protected_variable, transform_func)
    Z = stats.norm.ppf(level)
    lower_value = T_n - sigma_scalar * Z / np.sqrt(len(target_label))
    upper_value = T_n + sigma_scalar * Z / np.sqrt(len(target_label))
    return T_n, lower_value, upper_value