# PyTorch

For useful things learned while using torch.

### General

- cuda version incompatible with torch after upgrading torch: I just run `sudo apt-get -y install cuda` on a 64bit Ubuntu 16.04 and rebooted, worked after.
  - checked via `torch.cuda.is_available()`

- `torch.clamp(min, max)` - makes all values within a tensor that were under min or above max be min or max, respectively. Great for adding random noise but wanting to keep data within some boundaries, e.g.

```
mock_sigmoid_preds = mock_sigmoid_preds + (0.1**0.5)*torch.randn(mock_sigmoid_preds.size())
mock_sigmoid_preds = mock_sigmoid_preds.clamp(0.0, 1.0)
```


- `torch.Tensor.requires_grad` is the argument that specifies whether gradients are tracked for this tensor node. If requires_grad is True, then this tensor should also have a `grad_fn` attribute, which specifies the mathematical operator that created the variable (used in backwards pass). requires_grad is contagious, meaning that tensors that are transformations of the initial tensor will automatically also require and track gradients. `.detach()` is used to remove a tensor from the computation graph (gradients won't be tracked). Nicely explained [here](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/).

- `torch.autograd.set_detect_anomaly(True)` for debugging loss functions when e.g. sigmoid values end in strange range.
  - useful, but in the end either adding a requires_grad to one tensor in the custom loss or upgrading to torch==1.10.0 helped.

- `torch.full(size, fill_value)` - creates a tensor of specific size / dimensions, full of one given value.

- `tensor.requires_grad = False` makes the tensor (e.g. weights of a layer) frozen, not be trained.

### Count Model Parameters

```
def count_params(model, return_string=False):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = '{:,}'.format(params)
    if return_string:
        return params, 'The model has {} trainable parameters'.format(params)
    else:
        print('The model has {} trainable parameters'.format(params))
```

### CUDA / GPU Check for Tensors

- `tensor.is_cuda` gives a Boolean value whether a tensor is on a GPU or not.
  - to move a tensor to the preferred device, use `a_tensor = a_tensor.to(torch.device('cuda:0'))`
- `next(model.parameters()).is_cuda` is a quick way to check if the model is on the CUDA device
- alternatively tou can ask explicit for which device a thing is on `next(network.parameters()).device`
  - `tensor.device` works as well.

### CUDA / GPU Reload model onto CPU

When reloading a model that was saved on a GPU via model.save(), you have to add the map: `model = torch.load(path, map_location=torch.device('cpu'))`

### Truncating Tensors

If you know that you want your tensor to be truncated to a certain size, along each dimension, you can use numpy-like indexing to do it:

```
y = torch.rand(1, 100, 200)

# example limits
lim_dim_1 = 50
lim_dim_2 = 150

y_hat = y[:, :lim_dim_1, :lim_dim_2]
print(y_hat.size())
# should return (1, 50, 150)

```

These **upper limits are INCLUSIVE, unlike in Python lists**. You can also use torch's `torch.narrow()` to limit one dimension at a time, explained [here](https://pytorch.org/docs/stable/generated/torch.narrow.html), in the docs.

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
