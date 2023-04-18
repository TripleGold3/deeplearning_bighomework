import torch

from torch.autograd import Function as Function
import torch.nn.modules as nn
import torch.nn.functional as F
from torch.autograd import Variable
import MyBatchNorm1D_CPP

from MyBatchNormalization1D import MyBatchNormalization1d, MyBatchNormalization1dFunction
class MyBatchNormalization1dFunction_CPP(Function):
    @staticmethod
    def forward(ctx, X, train_flag, epsilon, gamma, beta, monmentum, moving_mean, moving_var):
        if train_flag:

            output, batch_norm, batch_mean, batch_var, batch_std, moving_mean, moving_var = \
                MyBatchNorm1D_CPP.train_forward(X, gamma, beta, moving_mean, moving_var, monmentum, epsilon)
            ctx.save_for_backward(X, batch_norm, batch_mean, batch_var, batch_std, gamma, beta, epsilon)
        else:
            output = MyBatchNorm1D_CPP.validation_forward(X, gamma, beta, moving_mean, moving_var, epsilon)
        return ctx, output
        
    @staticmethod
    def backward(ctx, grad_output):
        input, X_norm, batch_mean, batch_var, batch_sqrt_var, gamma, beta, epsilon = ctx.saved_tensors
        grad_input, grad_gamma, grad_beta = \
            MyBatchNorm1D_CPP.train_backward(grad_output, input, X_norm, batch_mean, batch_var, batch_sqrt_var, gamma, beta, epsilon)    
        return grad_input, None, None, grad_gamma, grad_beta, None, None, None



# 1D BatchNormlization
class MyBatchNormalization1d_CPP(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,device=None, dtype=None):
        super(MyBatchNormalization1d_CPP, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_features = num_features
       
        self.gamma = torch.nn.Parameter(torch.ones(num_features))
        self.beta = torch.nn.Parameter(torch.zeros(num_features))
        self.register_buffer('epsilon', eps * torch.ones(1, **factory_kwargs))
        self.register_buffer('momentum', momentum * torch.ones(1, **factory_kwargs))
        self.register_buffer('moving_mean', torch.zeros(num_features, **factory_kwargs))
        self.register_buffer('moving_var', torch.ones(num_features, **factory_kwargs))
    
        self.epsilon: Optional[Tensor]
        self.momentum: Optional[Tensor]
        self.moving_mean: Optional[Tensor]
        self.moving_var: Optional[Tensor]
      
    def forward(self, X):
        ctx,output = MyBatchNormalization1dFunction_CPP.apply(X, self.training, self.epsilon, self.gamma, self.beta, self.momentum, self.moving_mean, self.moving_var)
        return ctx,output
         
if __name__ == '__main__':
    # test
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    # input
    X = torch.randn(4, 3).to(device)
    X = Variable(X, requires_grad=True)


    # model
    model = MyBatchNormalization1d_CPP(3)
    model.to(device)


    # model.eval()
    ctx, output = model(X)
    print("cpp output:",output)
   
    # backward
    grad_output = torch.ones((4,3), requires_grad=True).to(device)
    loss = MyBatchNormalization1dFunction_CPP.backward(ctx, grad_output)
    print(loss)
    
    
    X_baseline = X.clone()
    X_baseline = Variable(X_baseline, requires_grad=True)
    model_baseline = MyBatchNormalization1d(3)
    model_baseline.to(device)
    ctx_baseline, output_baseline = model_baseline(X_baseline)
    print("python output:",output_baseline)  
    grad_output_baseline = torch.ones((4,3), requires_grad=True).to(device)
    loss_baseline = MyBatchNormalization1dFunction.backward(ctx_baseline, grad_output_baseline)
    print(loss_baseline)
   