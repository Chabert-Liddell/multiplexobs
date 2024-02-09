import torch
from multiplexobs.functional import log_norm_const

def test_log_norm_const():

    # Test case 1: Input tensor with all elements close to 0.5
    x = torch.tensor([0.5, 0.49])
    assert torch.allclose(log_norm_const(x[0]), log_norm_const(x[1]))


    # Additional test cases...