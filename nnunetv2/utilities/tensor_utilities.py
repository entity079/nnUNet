import torch

def sum_tensor(inp: torch.Tensor, axes, keepdim=False):
    """
    Sums a tensor over multiple axes.
    
    Parameters:
        inp (torch.Tensor): input tensor
        axes (list or tuple): axes to sum over
        keepdim (bool): whether to keep dimensions
    
    Returns:
        torch.Tensor
    """
    axes = list(set(axes))
    axes.sort()
    
    for ax in axes[::-1]:  # reverse order to avoid shifting indices
        inp = inp.sum(int(ax), keepdim=keepdim)
    
    return inp
