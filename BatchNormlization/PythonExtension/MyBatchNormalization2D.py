import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function as Function

class MyBatchNormalization2dFunction(Function):
    @staticmethod
    def forward(ctx, X, train_flag, epsilon, gamma, beta, monmentum, moving_mean, moving_var):
        if train_flag:
            batch_mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            batch_var = (X - batch_mean).pow(2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            batch_sqrt_var = torch.sqrt(batch_var + epsilon.unsqueeze(-1).unsqueeze(-1))
            X_norm = (X - batch_mean) / batch_sqrt_var
            output = gamma.unsqueeze(-1).unsqueeze(-1) * X_norm + beta.unsqueeze(-1).unsqueeze(-1)
            ctx.save_for_backward(X, X_norm, batch_mean, batch_var, batch_sqrt_var, gamma, beta, epsilon)
            moving_mean = (monmentum * moving_mean).unsqueeze(-1).unsqueeze(-1) + (1 - monmentum) * batch_mean
            moving_var = (monmentum * moving_var).unsqueeze(-1).unsqueeze(-1) + (1 - monmentum) * batch_var
        else:
            X_norm = (X - moving_mean.unsqueeze(-1).unsqueeze(-1)) / torch.sqrt((moving_var + epsilon).unsqueeze(-1).unsqueeze(-1))
            output = gamma.unsqueeze(-1).unsqueeze(-1) * X_norm + beta.unsqueeze(-1).unsqueeze(-1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, X_norm, batch_mean, batch_var, batch_sqrt_var, gamma, beta, epsilon = ctx.saved_tensors
        N = (grad_output.size(0))
        grad_beta = torch.sum(grad_output, dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        grad_gamma = torch.sum(grad_output * X_norm, dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        grad_X_norm = grad_output * (gamma.unsqueeze(-1).unsqueeze(-1))
        grad_X = (1. / N) * (1. / batch_sqrt_var) * (N * grad_X_norm - torch.sum(grad_X_norm, dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) - X_norm * torch.sum(grad_X_norm * X_norm, dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True))
        return grad_X, None, None,None, None, grad_gamma, grad_beta, None

class MyBatchNormalization2d(nn.Module):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1, dtype=None, device=None):
        super(MyBatchNormalization2d, self).__init__()
        factory_kwargs = {'dtype': dtype, 'device': device}
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones((1,num_features)))
        self.beta = nn.Parameter(torch.zeros((1,num_features)))
      
        self.register_buffer('epsilon', epsilon * torch.ones((1,num_features), **factory_kwargs))
        self.register_buffer('momentum', momentum * torch.ones(1, **factory_kwargs))
        self.register_buffer('moving_mean', torch.zeros((1,num_features), **factory_kwargs))
        self.register_buffer('moving_var', torch.ones((1,num_features), **factory_kwargs))
        self.epsilon: Optional[Tensor]
        self.momentum: Optional[Tensor]
        self.moving_mean: Optional[Tensor]
        self.moving_var: Optional[Tensor]
        
    def forward(self, X):
        return MyBatchNormalization2dFunction.apply(X, self.training, self.epsilon, self.gamma, self.beta, self.momentum, self.moving_mean, self.moving_var)
    
    
if __name__ == '__main__':
    # test
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # input
    X = torch.randn(4, 3, 2, 2).to(device)
    X = Variable(X, requires_grad=True)
    
    # model
    model = MyBatchNormalization2d(3).to(device)
    # model.eval()
    output = model(X)
    print(output)
    
   
    
    # backward
    loss = torch.sum(output)
    loss.backward()
    print(X.grad)
    


