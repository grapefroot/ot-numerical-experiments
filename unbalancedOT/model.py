import torch
import torchvision


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
            torch.nn.Linear(self.hidden_dims, self.features_out)
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
            torch.nn.Linear(self.hidden_dims, 1)
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
            torch.nn.Linear(self.hidden_dims, self.features_out)
        )

    def forward(self, *input) -> torch.Tensor:
        return self.omega(input[0])


class UnbalancedLoss:
    def __init__(self, norm_const1: int, norm_const2: int, distance_function, mass_variation_function, dual_function):
        self.norm_const1 = norm_const1
        self.norm_const2 = norm_const2
        self.distance_function = distance_function
        self.mass_variation_function = mass_variation_function
        self.dual_function = dual_function

    def compute(self, x, z, y, t_output, xi_output, f_output, f_t_output):
        result = self.norm_const1 * self.distance_function(x, t_output) * xi_output + \
                 self.norm_const1 * self.mass_variation_function(xi_output) + \
                 self.norm_const1 * xi_output * f_t_output - self.norm_const2 * self.dual_function(f_output)
        return result.mean()


def KL_dual(s):
    return torch.exp(s)


def Pearson_xi(s):
    return s ** 2 / 4 + s


def Hellinger(s):
    return s / (1 - s)


def Jensen_Shannon(s):
    # TODO: more numerically stable computation here
    return -torch.log(2 - torch.exp(s))


# T = TMapper(10, 5)
# Xi = PhiMapper(10, 1)
# f = OmegaMapper(5, 1)
# loss = UnbalancedLoss(10, 5)
# w_optim = torch.optim.SGD(f.parameters(), lr=-1e-2)
# txi_optim = torch.optim.SGD((T.parameters(), Xi.parameters()), lr=1e-2)
#
# def train_loop(data_loader: torch.utils.data.DataLoader):
#     for X, Z, Y in data_loader:
#         w_optim.zero_grad()
#         txi_optim.zero_grad()
#         loss_value = loss.compute(X, Z, Y, T(X, Z), Xi(X), f(Y), f(T(X, Z)))
#         loss_value.backward()
#         w_optim.step()
#         txi_optim.step()

