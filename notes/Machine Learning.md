---
tags: [Notebooks/Skills/Tech]
title: Machine Learning
created: '2021-11-19T16:02:40.897Z'
modified: '2021-11-20T14:49:20.662Z'
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

