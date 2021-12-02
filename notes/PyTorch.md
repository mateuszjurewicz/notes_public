---
title: PyTorch
created: '2021-11-25T14:14:41.382Z'
modified: '2021-12-02T10:40:09.096Z'
---

# PyTorch

For useful things learned while using torch.

### General

- `torch.full(size, fill_value)` - creates a tensor of specific size / dimensions, filled with one given value.

### Custom Loss Functions

Are easy to make, it's just that everything has to be in tensors, e.g. here's a dummy nn.MSELoss reimplementation:

```
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
```
