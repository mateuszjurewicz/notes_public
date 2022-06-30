# PyTorch

For useful things learned while using torch.

## General

- cuda version incompatible with torch after upgrading torch: I just run `sudo apt-get -y install cuda` on a 64bit Ubuntu 16.04 and rebooted, worked after.
  - checked via `torch.cuda.is_available()`

- `torch.clamp(input, min, max)` - makes all values within a tensor that were under min or above max be min or max, respectively. Great for adding random noise but wanting to keep data within some boundaries, e.g.

```
mock_sigmoid_preds = mock_sigmoid_preds + (0.1**0.5)*torch.randn(mock_sigmoid_preds.size())
mock_sigmoid_preds = mock_sigmoid_preds.clamp(0.0, 1.0)
```

- `torch.Tensor.round()` will round the elements of the tensor to nearest integer.

- `torch.Tensor.requires_grad` is the argument that specifies whether gradients are tracked for this tensor node. If requires_grad is True, then this tensor should also have a `grad_fn` attribute, which specifies the mathematical operator that created the variable (used in backwards pass). requires_grad is contagious, meaning that tensors that are transformations of the initial tensor will automatically also require and track gradients. `.detach()` is used to remove a tensor from the computation graph (gradients won't be tracked). Nicely explained [here](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/).

- `torch.autograd.set_detect_anomaly(True)` for debugging loss functions when e.g. sigmoid values end in strange range.
  - useful, but in the end either adding a requires_grad to one tensor in the custom loss or upgrading to torch==1.10.0 helped.

- `torch.full(size, fill_value)` - creates a tensor of specific size / dimensions, full of one given value.

- `tensor.requires_grad = False` makes the tensor (e.g. weights of a layer) frozen, not be trained.

## Index Select vs [:, inds.long(), :]
You may have a tensor and a list of indices along one of its dimension, that you want to grab. To do this, use `torch.index_select()`. Contrast it with masked_select, which uses a boolean tensor for the same task. **However, a quicker way is to use numpy-style indexing**. However, the former requires an int() type, the latter a long() type.

```
b = 10
n = 5
e = 4

data = torch.rand(b, n, e)

example_indices = torch.Tensor(
    [3, 5, 6, 7, 9]
).int()

r = torch.index_select(data, dim=0, index=example_indices)

# version via index_select()
print(r.size())
print(r)

# version via indexing:
print(data[example_indices.long(), :, :])
```

## Padding Tensors
The basic way to pad tensors is to use the `torch.nn.functional.pad()` function. However, this requires knowing by how much you want to pad (as opposed to "to what target size"). The way you construct padding instructions also differes depending on how many dimensions of the input tensor you want to pad. Here's an example for padding along 1 dimension (the second of two):

```
import torch
import torch.nn.functional as F

# input is (2, 3)
input = torch.Tensor(
    [
        [1, 1, 1],
        [2, 2, 2]
    ]
)

# we want our output to be (2, 5)
padding_left = 0
padding_right = 2
padding_token = -999

# if only 2 elements in padding instructions, they mean how much to pad on the left ad right
padding_instructions = (padding_left, padding_right)

out = F.pad(input, padding_instructions, "constant", padding_token)

print(out.size())  # (2, 3 + 2)
print(out)  # e.g. [1, 1, 1, -999, -999] ...
```

## Flatten Specific Dimensions
Let's say you have a data input tensor, which is a batch of sets of elements with some embedded dimensions, e.g. `data = torch.Tensor(batch=64, set_cardinality=100, elem_embed=2)`. but you want to flatter the batch and cardinality to get a tensor of size (64 * 100 = 6400, 2). You can use the `tensor.flatten()` function for this:

```
import torch

b = 64
c = 100
e = 2

input = torch.rand([b, c, e])
print(input.size())  # [64, 100, 2]

# flatten choosing the starting and end dimensions (to be flattened into a single one)
flattened = input.flatten(0, 1)
print(flattened.size())  # [6400, 2]
```

## Repeat / Tile a tensor into new dimensions
If you e.g. have a set representation that you want to concatenate to each end every set element's representation, you'll need to essentially copy it as many times as there are elements (the set's cardinality). You can do this using `torch.tile()`

```
cardinality = 20
elements = torch.rand(64, cardinality, 128)
set = torch.rand(64, 1, 128)

# repeate the set as many times as there are elements
# the 2nd argument is the number of repetitions per dimension
set_repeated = torch.tile(set, (1, cardinality, 1))

# now we can concat
es = torch.cat([elements, set_repeated], dim=2)

# es is now (64, 20, 128 + 128 = 256)
```

## Repeat / Tile a boolean tensor into a new dimension
I run into a situation where I needed a boolean mask tensor to be of the same size as my batched, encoded data (b, n_elem, emb_dim), so that I can elementwise-multiply the batch by the mask to zero out the elements I needed to have no influence on later calculations.

So I had `mask.size() = (b, n_elem)` and `enc_data.size() = (b, n_elem, emb_dim)`. I needed the mask to repeat its boolean value at the `n_elem` dimension into the new dimension, taken by the `emb_dim`.

```
anchors = torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
]).bool()

# we have 4 elements, batch of 2
n_elem = 4
b = 2
emb_dim = 5

# first, we need to create the new dimension for the mask's emb_dim
anchors = anchors.unsqueeze(2)

# then we use tile() to repeat the corresponding bool value into that entire dimension, per element
anchors = torch.tile(anchors, (1, 1, emb_dim))

# confirm
print(anchors.size())  # torch.Size([2, 4, 5])
print(anchors)

# tensor([[[ True,  True,  True,  True,  True],
#          [False, False, False, False, False],
#          [False, False, False, False, False],
#          [False, False, False, False, False]],
# 
#         [[False, False, False, False, False],
#          [ True,  True,  True,  True,  True],
#          [False, False, False, False, False],
#          [False, False, False, False, False]]])

```


## Create a Tensor with Random Values in Range (uniformly)
You sometimes want to have floating point values in a different range than between 0 and 1, preventing you from using `torch.rand()` and `torch.randint()` directly. You can use `torch.Tensor.uniform_()` instead:

```
# desired params
shape = (1, 5)
lower = 0
upper = 10

# generate
r = torch.FloatTensor(*shape).uniform_(lower, upper)

print(r)
# tensor([[2.6093, 0.6849, 5.6221, 4.1591, 9.7290]])
```

## Find Indices in Tensor Based on Filter Condition
You may want to get the indices of elements of a tensor that fulfill a certain condition. This is done through direct comparison and the `torch.Tensor.nonzero()` function.

```
# random tensor
r = torch.rand((2, 10))
above_point_five_indices = (r > 0.5).nonzero()
```
Note that in this case the indices are themselves pointing to both dimensions of the original matrix.

## Filter to get Boolean Mask
You can also use a tensor's built-in `torch.Tensor.ge()` function, which will return a boolean tensor of the same shape as the original tensor, with True of False at indices whose elements meet or break the condition:

```
inp = torch.Tensor([
    [1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9]
])

bool_mask = inp.ge(5)
print(bool_mask)

# tensor([[False, False, False],
#         [False,  True,  True],
#         [ True,  True,  True]])
```
**We can then use the mask via `bool_mask.nonzero()` to get the indices!**

And select those elements that meet the condition via `torch.masked_select()`:

```
(...)
elements_that_meet_the_condition = torch.masked_select(inp, bool_mask)

print(elements_that_meet_the_condition)
# tensor([5., 6., 7., 8., 9.])
```

## Filter an encoded batch via boolean mask
I run into a situation where I needed a boolean mask tensor (b, n) to be of the same size as my batched, encoded data (b, n, e), so that I can get only the examples from the batch for which the boolean mask was True (b, n_True, e).

 One approach was to use loops over the matrix in a list comprehension and pass it to torch.stack(), but flattening turned out much more simple. **The one requirement was that in for each example, the boolean mask has the same amount of True values**, thus guaranteeing that the final reshape is possible (wouldn't result in a jagged array). So n_True same for each example in batch.

### Setup
```
# boolean mask (b, n) over elements of each example
# same amount of True in each is a HARD REQUIREMENT
unassigned = torch.Tensor([
    [True, True, False],
    [True, False, True]
]).bool()

# the above requires the batch size to be 2, and the n of elems to be 3
b = unassigned.size(0)  # 2
n = unassigned.size(1)  # 3
e = 5  # embedding size, per element

# mock data
enc_data = torch.rand((b, n, e))
```

### Loop approach:

```
# 1. LOOP APPROACH

# this gives us a (b, n_True) tensor, where n_True is the number of True in boolean mask's row
unassigned_indices = torch.stack([r.nonzero().flatten(0, 1) for r in unassigned], dim=0)

# use indices to grab elements for each row in the data
unassigned_encoded_data = torch.stack([enc_data[i][inds] for i, inds in enumerate(unassigned_indices)])

# show & confirm
print(enc_data)
print(unassigned_encoded_data.size())  # should be (b, n_True, e)
print(unassigned_encoded_data)
```

### Flatten approach:

```
# 2. FLATTEN APPROACH

# first we'll need the mask extended into the new dimension
unassigned_mask = unassigned.unsqueeze(2)  # (b, n, 1)
unassigned_mask = torch.tile(unassigned_mask, (1, 1, e))  # (b, n, e)

# flatten mask
unassigned_mask = torch.flatten(unassigned_mask, 0, 2)  # (b * n * e)

# flatten data
flat_data = torch.flatten(enc_data, 0, 2)  # (b * n * e)

# masked select
unassigned_flat_data = torch.masked_select(flat_data, unassigned_mask)  # (b * n_True * e)

# reshape
unassigned_enc_data = torch.reshape(unassigned_flat_data, (b, n_True, e)) # (b, n_True, e)

# confirm
print(enc_data)
print(unassigned_enc_data.size())
print(unassigned_enc_data)
```

## Count Model Parameters

```
def count_params(model, return_string=False):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = '{:,}'.format(params)
    if return_string:
        return params, 'The model has {} trainable parameters'.format(params)
    else:
        print('The model has {} trainable parameters'.format(params))
```

## CUDA / GPU Check for Tensors

- `tensor.is_cuda` gives a Boolean value whether a tensor is on a GPU or not.
  - to move a tensor to the preferred device, use `a_tensor = a_tensor.to(torch.device('cuda:0'))`
- `next(model.parameters()).is_cuda` is a quick way to check if the model is on the CUDA device
- alternatively tou can ask explicit for which device a thing is on `next(network.parameters()).device`
  - `tensor.device` works as well.

## CUDA / GPU Reload model onto CPU

When reloading a model that was saved on a GPU via model.save(), you have to add the map: `model = torch.load(path, map_location=torch.device('cpu'))`

## Truncating Tensors

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

## Stack or Concatenate

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

## Custom Loss Functions

Are easy to make, it's just that everything has to be in tensors, e.g. here's a dummy nn.MSELoss reimplementation:

```
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
```

## Freezing Layers

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
