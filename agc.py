import torch
import warnings
from torch._six import inf
from typing import Union, Iterable

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def AGC(parameters: _tensor_or_tensors, clip: float = 1e-3, eps: float = 1e-3, zero_division_eps: float = 1e-6):
    """Adaptively clips gradients of an iterable of parameters.
    
    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip: (float) Maximum allowed ratio of update norm to parameter norm.
        eps: (float) epsilon term to prevent clipping of zero-initialized params.
        zero_division_eps: (float) epsilon term to prevent division by zero.
    
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    for p in parameters:
        if p.grad is None:
            continue
            
        g_norm = unitwise_norm(p.grad, zero_division_eps)
        p_norm = unitwise_norm(p, eps)
        
        trigger = (g_norm / p_norm) > clip
        norm_divergence = p_norm / g_norm * clip
        
        grad_scale = torch.where(trigger, norm_divergence, torch.ones_like(g_norm))
        
        p.grad.data.copy_(p.grad * grad_scale)
    
    
def unitwise_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
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
        raise ValueError(f'Got a parameter with ndims not in 0-4! {x}')
    return x.norm(2, dim, keepdims).max(eps)
