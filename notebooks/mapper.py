import numpy as np

def construct_map(weights, data_first, data_second, coupling):
    weight_1, weight_2 = weights
 #   print(weights)
    n0 = data_first.shape[0]
    n1 = data_second.shape[0]
    mapped_class_0 = weight_1 * data_first  + n0 * weight_2 * (coupling @ data_second)
    mapped_class_1 = weight_1 * n1 * (coupling.T @ data_first) + weight_2 * data_second
    return mapped_class_0, mapped_class_1

def full_repair(data_first, data_second, coupling, y_first=None, y_second=None, weights=[0.5, 0.5]):
    mapped_class_0, mapped_class_1 = construct_map(weights, data_first, data_second, coupling)
    if y_first is None or y_second is None:
        return np.concatenate((mapped_class_0, mapped_class_1))
    else:
        return np.concatenate((mapped_class_0, mapped_class_1)), np.concatenate((y_first, y_second))

def partial_repair(interpolation_weight, data_first, data_second, coupling, y_first=None, y_second=None, weights=[0.5, 0.5]):
    mapped_class_0, mapped_class_1 = construct_map(weights, data_first, data_second, coupling)
    if y_first is None or y_second is None:
        return np.concatenate(
            (interpolation_weight * mapped_class_0 + (1 - interpolation_weight) * data_first,
             interpolation_weight * mapped_class_1 + (1 - interpolation_weight) * data_second))
    else:
        return np.concatenate(
            (interpolation_weight * mapped_class_0 + (1 - interpolation_weight) * data_first,
             interpolation_weight * mapped_class_1 + (1 - interpolation_weight) * data_second)), np.concatenate((y_first, y_second))

def random_repair(interpolation_weight, theta, data_first, data_second, coupling, y_first=None, y_second=None, weights=[0.5, 0.5]):
    mapped_class_0, mapped_class_1 = construct_map(weights, data_first, data_second, coupling)
    selector_0 = np.random.binomial(1, theta, size=data_first.shape[0]) == 1
    selector_1 = np.random.binomial(1, theta, size=data_second.shape[0]) == 1
    
    data_first[selector_0] = interpolation_weight * mapped_class_0[selector_0] + (1 - interpolation_weight) * data_first[selector_0]
    data_second[selector_1] = interpolation_weight * mapped_class_1[selector_1] + (1 - interpolation_weight) * data_second[selector_1]
    
    if y_first is None or y_second is None:
        return np.concatenate(
            (interpolation_weight * mapped_class_0 + (1 - interpolation_weight) * data_first,
             interpolation_weight * mapped_class_1 + (1 - interpolation_weight) * data_second))
    else:
        return np.concatenate(
            (interpolation_weight * mapped_class_0 + (1 - interpolation_weight) * data_first,
             interpolation_weight * mapped_class_1 + (1 - interpolation_weight) * data_second)), np.concatenate((y_first, y_second))
