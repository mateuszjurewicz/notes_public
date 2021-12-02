---
tags: [Notebooks/Skills/Tech]
title: Machine Learning
created: '2021-11-19T16:02:40.897Z'
modified: '2021-12-02T10:37:55.243Z'
---

# Machine Learning

Definitions of concepts, practical tips, possibly notes on select papers. Goal is to solidify understanding by writing things down / explaining them to myself.


### Chain Rule

In calculus the chain rule is a formula for calculating the derivative of the composition $f(g(x)) = (f \circ g)(x)$ of two differentiable functions $f$ and $g$.
Specifically, the chain rule is: $\frac{df}{dx} = \frac{df}{dg} \times \frac{dg}{dx}$

You can think of this more intuitively through this example. Knowing the rate of change of $f$ relative to $g$ and the rate of change of $g$ relative to $x$, we can calculate the rate of change of $f$ relative to $x$ by multiplying the former two values. 

Specifically, if a car travels 2x as fast as a bike, and a bike travels 4x faster than a human, then a car travels 2x4=8 times faster than a human.

The Leibniz notation of $\frac{df}{dx}$ can be read as _the direction and rate of change of the value of $f$ as $x$ is changing_.

Chain Rule is often used e.g. in RNNs and RL, as at each time step we go back to some past time step to calculate current loss function. We're looking for the derivative of the parameters with respect to the value of the loss function $\frac{d \theta}{d L}$.


### PonderNet

Is a more recent improvement of the Alex Graves' ACT (adaptive computation time). It's a way for the model to adjust the number of computation steps to the input.

A Yannic Kilcher video explaining the paper:
https://www.youtube.com/watch?v=nQDZmf2Yb9k

A good code implementation via github:
- https://nn.labml.ai/adaptive_computation/ponder_net/index.html
  - from https://github.com/labmlai/annotated_deep_learning_paper_implementations
- https://github.com/lucidrains/ponder-transformer


### Reparametrization trick

Is about being able to backpropagate the gradients back through a **variational** autoencoder (as opposed to just a normal autoencoder which generates a latent vector representation of the input and not a sequence of means and standard deviations per latent component). We can't backpropagate through a sampling operation, which is what happens when the VAE obtains an actual latent vector representation from the learned distribution of latent components.

It takes advantage of the property of the Gaussian distribution, such that if there's $\mathcal{N}_1(0,1)$ and another $\mathcal{N}_2(\mu, \sigma)$, then if we sample from the first one and get $X_1$ and from the second one $X_2$, we also know that $X_2 = X_1 * \sigma + \mu$.

So since the VAE learns a $\mu$ and $\sigma$ we can backpropagate throught that formula above, not touching the sampling.

Interestingly explained here: 
https://youtu.be/EeMhj0sPrhE?t=1178

And very simple VAE in code:
https://github.com/pytorch/examples/blob/master/vae/main.py 

### Activation functions

Nonlinear functions often applied as the last tranformation in a neural network layer, giving them greater representation power than just a linear transform. In terms of biological inspiration, they are supposed to mimic the action potential of neurons (i.e. fire or don't fire, past a threshold). They usually have to be differentiable to allow for gradient-based learning.

Common activation functions:
- `ReLU` - rectified linear activation unit. Everything below zero gets turned to zero, everything above stays itself.
$\textrm{ReLU}(x) = \textrm{max}(0.0, x)$

- `Sigmoid` - aka `Logistic`, everything gets pushed between 0 and 1, with a hyperbolic-like curve in the middle. Most things below -5 and above 5 get pushed to almost -1 and almost 1. Recommended to use the Xavier Glorot's `Xavier Uniform` weight initialization and scale input data to 0-1 when using sigmoid.
$\textrm{Sigmoid}(x) = 1.0~/~(1.0 + e^{-x})$

- `TanH` - aka `hyperbolic tangent` function. Same shape as Sigmoid, but it has the range between -1 and 1.
$\textrm{TanH}(x) = (e^x – e^{-x})~/~(e^x + e^{-x})$

- `Softmax` - pushes a single value to be high whilst everything else in the input becomes lower and turns them into a proper probability vector (summing to 1).
$\textrm{Softmax}(x) = e^x / \textrm{sum}(e^x)$

- `Swish` - less known, developed by google, supposed to be good for deeper models. Graph looks much like ReLU but the formula is like Sigmoid, values range from slightly negative to infinity.
$\textrm{Sigmoid}(x) = x~/~(1.0 + e^{-x}) = x * \textrm{Sigmoid}(x)$
