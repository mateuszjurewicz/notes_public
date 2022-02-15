---
title: PyTorch
created: '2021-11-25T14:14:41.382Z'
modified: '2022-02-15T08:45:30.428Z'
---

# PyTorch

For useful things learned while using torch.

### General

- `torch.Tensor.requires_grad` is the argument that specifies whether gradients are tracked for this tensor node. If requires_grad is True, then this tensor should also have a `grad_fn` attribute, which specifies the mathematical operator that created the variable (used in backwards pass). requires_grad is contagious, meaning that tensors that are transformations of the initial tensor will automatically also require and track gradients. `.detach()` is used to remove a tensor from the computation graph (gradients won't be tracked). Nicely explained [here](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/).
- `torch.autograd.set_detect_anomaly(True)` for debugging loss functions when e.g. sigmoid values end in strange range.
- `torch.full(size, fill_value)` - creates a tensor of specific size / dimensions, full of one given value.
- `tensor.is_cuda` gives a Boolean value whether a tensor is on a GPU or not.
- `tensor.requires_grad = False` makes the tensor (e.g. weights of a layer) frozen, not be trained.

### Stack or Concatenate

Often when creating sythetic training examples (`x` and `y`) I end up having to create them one by one in a loop and then put them together into a tensor of size batch or dataset length. 

To do this, first just append your tensors to a placeholder empty list and then either `torch.stack` or `torch.concatenate`. 

The difference comes from **cat** using given dimension, and **stack** concatenating them along a new dimension.

```
a = torch.rand(3, 4)
b = torch.rand(3, 4)

catted = torch.cat([a, b], dim=0)  # (6, 4)
stacked = torch.stack([a, b], dim=0)  # (2, 3, 4)
```

So it depends if you created individual examples already unsqueezed or not. 

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
