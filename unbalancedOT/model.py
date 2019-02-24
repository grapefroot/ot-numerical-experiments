from collections import Counter

import numpy as np
import torch
import torch.utils.data
from typing import List


class UnbalancedSampler(torch.utils.data.Sampler):

    def __init__(self, data_source: torch.utils.data.Dataset, class_count: List[int], num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source

        if num_samples is not None:
            self.num_samples = num_samples
        else:
            self.num_samples = len(self.data_source)

        self.class_prob = np.array(class_count, dtype=np.float64) / np.sum(class_count)
        target_list = self.data_source.targets.tolist()
        self.counter = Counter(target_list)
        self.p = [self.class_prob[label] / self.counter[label] for label in target_list]

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.from_numpy(np.random.choice(np.arange(n), size=self.num_samples, p=self.p)).tolist())

    def __len__(self):
        return len(self.data_source)


class TMapper(torch.nn.Module):
    def __init__(self, features_in: int, features_out: int, hidden_dims=512):
        super(TMapper, self).__init__()
        self.features_in = features_in
        self.features_out = features_out
        self.hidden_dims = hidden_dims
        self.theta = torch.nn.Sequential(
            torch.nn.Linear(self.features_in + 1, self.hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims, self.hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims, self.hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims, self.features_out),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, *input):
        return self.theta(torch.cat(input, dim=1))


class PhiMapper(torch.nn.Module):
    def __init__(self, features_in: int, features_out: int, hidden_dims=512):
        super(PhiMapper, self).__init__()
        self.features_in = features_in
        self.features_out = features_out
        self.hidden_dims = hidden_dims
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(self.features_in, self.hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims, self.hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims, self.hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims, 1),
            torch.nn.Softplus()
        )

    def forward(self, *input):
        return self.phi(input[0])


class OmegaMapper(torch.nn.Module):
    def __init__(self, features_in: int, features_out: int, hidden_dims=512):
        super(OmegaMapper, self).__init__()
        self.features_in = features_in
        self.features_out = features_out
        self.hidden_dims = hidden_dims
        self.omega = torch.nn.Sequential(
            torch.nn.Linear(self.features_in, self.hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims, self.hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims, self.hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims, self.features_out),
        )

    def forward(self, *input) -> torch.Tensor:
        return self.omega(input[0])


class UnbalancedLoss:
    def __init__(self, norm_const1: float, norm_const2: float, distance_function, mass_variation_function,
                 dual_function):
        self.norm_const1 = norm_const1
        self.norm_const2 = norm_const2
        self.distance_function = distance_function
        self.mass_variation_function = mass_variation_function
        self.dual_function = dual_function

    def compute(self, x, z, y, t_output, xi_output, f_output, f_t_output, verbose=False):
        df_out = self.distance_function(x, t_output)
        mvf_out = self.mass_variation_function(xi_output)
        dual_out = self.dual_function(f_output)

        if verbose:
            print('Mass variation output {}'.format(mvf_out.mean()))
            print('Distance function output {}'.format(df_out.mean()))
            print('Dual function output {}'.format(dual_out.mean()))

            print('Terms:')
            print('First {}'.format((self.norm_const1 * df_out * xi_output).mean()))
            print('Second {}'.format((self.norm_const1 * mvf_out).mean()))
            print('Third {}'.format((self.norm_const1 * xi_output * f_t_output).mean()))
            print('Fourth {}'.format((-self.norm_const2 * dual_out).mean()))

        result = self.norm_const1 * df_out * xi_output + \
                 self.norm_const1 * mvf_out + \
                 self.norm_const1 * xi_output * f_t_output - self.norm_const2 * dual_out
        return result.mean()


def KL_dual(s):
    return torch.exp(s)


def Pearson_xi_dual(s):
    return s ** 2 / 4 + s


def Hellinger_dual(s):
    return s / (1 - s)


def Jensen_Shannon_dual(s):
    # TODO: more numerically stable computation here
    return -torch.log(2 - torch.exp(s))
