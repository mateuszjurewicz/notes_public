---
title: PyTorch
created: '2021-11-25T14:14:41.382Z'
modified: '2021-12-11T10:15:12.637Z'
---

# PyTorch

For useful things learned while using torch.

### General

- `torch.full(size, fill_value)` - creates a tensor of specific size / dimensions, full of one given value.
- `tensor.is_cuda` gives a Boolean value whether a tensor is on a GPU or not.
- `tensor.requires_grad = False` makes the tensor (e.g. weights of a layer) frozen, not be trained.

### Custom Loss Functions

Are easy to make, it's just that everything has to be in tensors, e.g. here's a dummy nn.MSELoss reimplementation:

```
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
```

### Freezing Layers

E.g. in a sentence ordering model, if I need to freeze the BERT layers (to avoid having to adjust an insane amount of params), I can make those layers' parameters not require gradient and also tell the optimizer to only touch the parameters that require grads:

```
for param in model.language_model.parameters():
  param.requires_grad = False

(...)

optimizer = optim.AdamW(
  filter(lambda p: p.requires_grad, model.parameters()),
  lr=0.001)
)
```
There may however be some nuance about batch normalization to look out for.

Source: https://androidkt.com/pytorch-freeze-layer-fixed-feature-extractor-transfer-learning/
