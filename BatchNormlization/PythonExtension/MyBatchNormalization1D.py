import torch

from torch.autograd import Function as Function
import torch.nn.modules as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MyBatchNormalization1dFunction(Function):
    @staticmethod
    def forward(ctx, X, train_flag, epsilon, gamma, beta, monmentum, moving_mean, moving_var):
        if train_flag:
            batch_mean = X.mean(dim=0, keepdim=True)
            batch_var = (X - batch_mean).pow(2).mean(dim=0, keepdim=True)
            batch_sqrt_var = torch.sqrt(batch_var + epsilon)
            X_norm = (X - batch_mean) / batch_sqrt_var
            output = gamma * X_norm + beta
            ctx.save_for_backward(X, X_norm, batch_mean, batch_var, batch_sqrt_var, gamma, beta, epsilon)
            moving_mean = monmentum * moving_mean + (1 - monmentum) * batch_mean
            moving_var = monmentum * moving_var + (1 - monmentum) * batch_var
        else:
            X_norm = (X - moving_mean) / torch.sqrt(moving_var + epsilon)
            output = gamma * X_norm + beta
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        input, X_norm, batch_mean, batch_var, batch_sqrt_var, gamma, beta, epsilon = ctx.saved_tensors
        N = (grad_output.size(0))
        grad_beta = torch.sum(grad_output, dim=0, keepdim=True)
        grad_gamma = torch.sum(grad_output * X_norm, dim=0, keepdim=True)
        grad_X_norm = grad_output * gamma
        grad_batch_var = torch.sum(grad_X_norm * (input - batch_mean) * (-0.5) * (batch_var + epsilon).pow(-1.5), dim=0, keepdim=True)
        #########这里的batch_sqrt_var是不是该考虑eppsilon的影响
        grad_batch_mean = torch.sum(grad_X_norm * (-1) / batch_sqrt_var, dim=0, keepdim=True) + grad_batch_var * torch.sum(-2 * (input - batch_mean), dim=0, keepdim=True) / N
        grad_input = grad_X_norm / batch_sqrt_var + grad_batch_var * 2 * (input - batch_mean) / N + grad_batch_mean / N
        return grad_input, None, None, None, None, grad_gamma, grad_beta, None

# 1D BatchNormlization
class MyBatchNormalization1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(MyBatchNormalization1d, self).__init__()
        self.num_features = num_features
        self.eps = eps * torch.ones(1, requires_grad=True)
        self.momentum = momentum * torch.ones(1, requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(num_features))
        self.beta = torch.nn.Parameter(torch.zeros(num_features))
        self.mean = torch.zeros(num_features,requires_grad=True)
        self.var = torch.ones(num_features,requires_grad=True)
    def forward(self, X):
        output = MyBatchNormalization1dFunction.apply(X, self.training, self.eps, self.gamma, self.beta, self.momentum, self.mean, self.var)
        return output
         
