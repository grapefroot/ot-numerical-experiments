import numpy as np
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from sklearn.metrics import accuracy_score, f1_score


def confusion_matrix(predictions, protected_attribute):
    n = predictions.shape[0]
    a = np.sum((predictions == 0) & (protected_attribute == 0)) / n
    b = np.sum((predictions == 0) & (protected_attribute == 1)) / n
    c = np.sum((predictions == 1) & (protected_attribute == 0)) / n
    d = np.sum((predictions == 1) & (protected_attribute == 1)) / n
    return a, b, c, d

def pretty_confusion_matrix(predictions, protected_attribute):
    a, b, c, d = confusion_matrix(predictions, protected_attribute)
    data = np.empty((2, 2))
    data[0, 0] = a
    data[0, 1] = c
    data[1 ,0] = b
    data[1, 1] = d
    return pd.DataFrame(data=data * predictions.shape[0], index=['C = 0', 'C = 1'], columns=['Y = 0', 'Y = 1'])

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


def subsample_maps(num_samples, num_couplings, data_first, data_second, y_first, y_second, coupling_function):
    assert num_samples < min(len(data_first, data_second))
    index_1 = np.random.choice(np.arange(0, len(data_first)), size=num_samples, replace=False)
    index_2 = np.random.choice(np.arange(0, len(data_second)), size=num_samples, replace=False)
    couplings = []
    for coupling_index in range(num_couplings):
        coupling = coupling_function(data_first[index_1], data_second[index_2], y_first[index_1], y_second[index_2])
        couplings.append(coupling)
    return couplings


def evaluate_metrics(y_true, y_predicted, protected_variable, fairness_metrics, accuracy_metrics, verbose=False):
    """
    y : np.array[n] - target variable or output of classifier
    protected_variable: np.array[n] - binary protected variable
    metrics: dict the value of metrics to evaluate.
    Each must take the y and protected_variable as arguments
    """
    predicted_fairness_dict = {metric_name : metric_func(y_predicted, protected_variable) for (metric_name, metric_func) in fairness_metrics.items()}
    target_fairness_dict = {'{}_target'.format(metric_name): metric_func(y_true, protected_variable) for (metric_name, metric_func) in fairness_metrics.items()}
    accuracy_dict = {metric_name: metric_func(y_true, y_predicted) for (metric_name, metric_func) in accuracy_metrics.items()}
    return {**predicted_fairness_dict, **target_fairness_dict, **accuracy_dict}


def evaluate_repair(conditional_0, conditional_1, y_first, y_second, protected_test, repair_funciton, clf, predicted_fairness_dict, accuracy_metrics_dict, name, n_trials = 1, threshold = 0.5, num=100, *args, **kwargs):
    """
    Name:str - name of the method to repair
    Returns:
    Dict containing a list for every metric
    """
    metrics_values = defaultdict(list)
    metrics_values['name'] = name
    for repair_value in tqdm(np.linspace(0.01, 1, num=num)):
        trial_container = defaultdict(list)
        for trial_index in range(n_trials):
            X_new, y_new = repair_funciton(repair_value, conditional_0, conditional_1, y_first, y_second)
            y_predicted = (clf.predict_proba(X_new)[:, 1] >= threshold).astype(np.int)
            metrics_value = evaluate_metrics(y_new, y_predicted, protected_test, predicted_fairness_dict, accuracy_metrics_dict)
            for key, value in metrics_value.items():
                trial_container[key].append(value)
        #average within each container    
        trial_container = {key:np.mean(value, axis=0) for (key, value) in trial_container.items()}
            
        for key, value in trial_container.items():
            metrics_values[key].append(value)
    return metrics_values


def plot_metrics(metric_dicts, what_to_plot=None, subplot_shape=None, subplot_size=None):
    """
    Metric dicts: List[dict] - list of dictionaries corresponding to metrics
    what_to_plot: List[str] - list of metric names
    """
    if not isinstance(metric_dicts, list):
        metric_dicts = [metric_dicts]
        
    if what_to_plot is None:
        what_to_plot = list(metric_dicts[0].keys() - ['name'])
        
    if not isinstance(what_to_plot, list):
        what_to_plot = [what_to_plot]
     
    if subplot_shape is None:
        nrows, ncols = len(what_to_plot), 1
    else:
        nrows, ncols = subplot_shape
        
    if subplot_size is None:
        figsize = (10, 10*len(what_to_plot))
    else:
        figsize = subplot_size
    
    with sns.axes_style('darkgrid'):    
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        
        for subplot_index, metric_name in enumerate(what_to_plot):
            for entry in metric_dicts:
                values = entry[metric_name]
                axes[subplot_index].set_title(metric_name)

                if metric_name.startswith('CI'):
                    tn, lower, upper = zip(*values)
                    axes[subplot_index].plot(np.linspace(0, 1, num=len(values)), tn, label=entry['name'])
                    axes[subplot_index].fill_between(np.linspace(0, 1, num=len(values)), upper, lower, alpha=0.5)
                else:    
                    axes[subplot_index].plot(np.linspace(0, 1, num=len(values)), values, label=entry['name'])
                axes[subplot_index].set_ylim(0, 1.5)
                axes[subplot_index].axhline(y=0.8, xmin=0, xmax=1, linestyle=':')
                axes[subplot_index].axhline(y=1.2, xmin=0, xmax=1, linestyle=':')
                axes[subplot_index].legend()
       # plt.show()
    return fig, axes 
    
    
def plot_pies(points, protected_variable, label, *args, **kwargs):
    plt.figure(figsize=(20, 20))
    
    r1 = 0.5
    r2 = r1 + 0.5

    size=100
    
    x = [0] + np.cos(np.linspace(0, 2*math.pi*r1, 10)).tolist()
    y = [0] + np.sin(np.linspace(0, 2*math.pi*r1, 10)).tolist()

    xy1 = list(zip(x, y))
    s1 = max(max(x), max(y))

    # ...
    x = [0] + np.cos(np.linspace(2*math.pi*r1, 2*math.pi*r2, 10)).tolist()
    y = [0] + np.sin(np.linspace(2*math.pi*r1, 2*math.pi*r2, 10)).tolist()
    
    xy2 = list(zip(x, y))
    s2 = max(max(x), max(y))

    fig, ax = plt.subplots(figsize=(10, 10))

    #algorithm1
    ax.scatter(points[:, 0], points[:, 1], marker=(xy1, 0),
               s=size, c=protected_variable, label='protected variable', *args, **kwargs)

    #algorithm2
    ax.scatter(points[:, 0], points[:, 1], marker=(xy2, 0),
               s=size, c=label, label='label', *args, **kwargs)

    plt.legend(loc='best', fontsize = 'x-large')
        
 #   plt.show()
    return fig, ax
    
def di_acc_curve(clf, data, labels, protected):
    probas = clf.predict_proba(data)[:, 1]
    accuracy = []
    di = []
    for thresh in np.linspace(0.1, 0.9, num=100):
        predictions =  (probas >= thresh).astype(np.int)
        accuracy.append(accuracy_score(labels, predictions))
        di.append(disparate_impact(predictions, protected))
    return accuracy, di