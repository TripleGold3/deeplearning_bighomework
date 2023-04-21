import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function as Function
import MyBatchNorm2D_CPP
class MyBatchNormalization2dFunction_CPP(Function):
    @staticmethod
    def forward(ctx, someclass,  X, train_flag, epsilon, gamma, beta, monmentum, moving_mean, moving_var):
        if train_flag:           
            output, batch_norm, batch_mean, batch_var, batch_std, moving_mean, moving_var = \
                MyBatchNorm2D_CPP.train_forward(X, gamma, beta, moving_mean, moving_var, monmentum, epsilon)
            someclass.moving_mean = moving_mean
            someclass.moving_var = moving_var
            ctx.save_for_backward(X, batch_norm, batch_mean, batch_var, batch_std, gamma, beta, epsilon)
        else:
            [output] = MyBatchNorm2D_CPP.validation_forward(X, gamma, beta, moving_mean, moving_var, epsilon)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, X_norm, batch_mean, batch_var, batch_sqrt_var, gamma, beta, epsilon = ctx.saved_tensors
        grad_input, grad_gamma, grad_beta = \
            MyBatchNorm2D_CPP.train_backward(grad_output, input, X_norm, batch_mean, batch_var, batch_sqrt_var, gamma, beta, epsilon)    
        return None, grad_input, None, None, grad_gamma, grad_beta, None, None, None 

class MyBatchNormalization2d_CPP(nn.Module):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1, dtype=None, device=None):
        super(MyBatchNormalization2d_CPP, self).__init__()
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
        output =  MyBatchNormalization2dFunction_CPP.apply(self if self.training else None, X, self.training, self.epsilon, self.gamma, self.beta, self.momentum, self.moving_mean, self.moving_var)
        return output
  
    

    


         
if __name__ == '__main__':
    # test
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    device = torch.device("cpu")
    # input
    X = torch.randn(4, 3,2,2).to(device)
    X = Variable(X, requires_grad=True)


    # model
    model = MyBatchNormalization2d_CPP(3)
    model.to(device)


    # model.eval()
    output = model(X)
    print("cpp output:",output)
   
    # backward
    loss = torch.sum(output)
    loss.backward(retain_graph=True)
    print(loss)
    
    
#     X_baseline = X.clone()
#     X_baseline = Variable(X_baseline, requires_grad=True)
#     model_baseline = MyBatchNormalization1d(3)
#     model_baseline.to(device)
#     ctx_baseline, output_baseline = model_baseline(X_baseline)
#     print("python output:",output_baseline)  
#     grad_output_baseline = torch.ones((4,3), requires_grad=True).to(device)
#     loss_baseline = MyBatchNormalization1dFunction.backward(ctx_baseline, grad_output_baseline)
#     print(loss_baseline)
   