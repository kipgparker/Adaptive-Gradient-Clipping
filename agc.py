import torch
import warnings
from torch._six import inf
from typing import Union, Iterable

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def AGC(parameters: _tensor_or_tensors, clip: float = 1e-3, eps: float = 1e-3):
    """Adaptively clips gradients of an iterable of parameters.
    
    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip: (float) Maximum allowed ratio of update norm to parameter norm.
        eps: (float) epsilon term to prevent clipping of zero-initialized params.
    
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    
    for p in parameters:
        clip_tensor = torch.tensor(clip).to(p.device) 
        eps_tensor = torch.tensor(eps).to(p.device) 

        g_norm = unitwise_norm(p.grad)
        p_norm = unitwise_norm(p)
        
        max_norm = clip_tensor * torch.max(p_norm, eps_tensor)
        p.grad.data.copy_(my_clip(g_norm, max_norm, p.grad))
    
    
def my_clip(g_norm: torch.Tensor, max_norm: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    trigger = g_norm < max_norm
    # This little max(., 1e-6) is distinct from the normal eps and just prevents
    # division by zero. It technically should be impossible to engage.
    small = torch.tensor(1e-6).to(g_norm.device)
    clipped_grad = grad * (max_norm / torch.max(g_norm, small))
    return torch.where(trigger, grad, clipped_grad)
        
    
def unitwise_norm(x: torch.Tensor) -> torch.Tensor:
    """Compute norms of each output unit separately, also for linear layers."""
    if x.ndim <= 1: # Scalars and vectors
        dim = 0
        keepdims = False
    elif x.ndim in [2, 3]: # Linear layers of shape IO or multihead linear
        dim = 0
        keepdims = True
    elif x.ndim == 4: # Conv kernels of shape IOHW
        # other code source uses dim = [0, 1, 2,], but i assume its for convolution order
        dim = [1, 2, 3]
        keepdims = True
    else:
        raise ValueError(f'Got a parameter with shape not in [1, 2, 4]! {x}')
    return compute_norm(x, dim, keepdims)


def compute_norm(x: torch.Tensor, dim: list, keepdims: bool) -> torch.Tensor:
    """Axis-wise euclidean norm."""
    return torch.sum(x ** 2, dim=dim, keepdims=keepdims) ** 0.5