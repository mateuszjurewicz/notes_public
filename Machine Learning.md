# Machine Learning

Definitions of concepts, practical tips, possibly notes on select papers. Goal is to solidify understanding by writing things down / explaining them to myself.


### Hierarchical Clustering

Is an **unsupervised** method to cluster data points based on their distance matrix. The hierarchy comes from doing it in steps, either starting with each point forming its own cluster (agglomerative) or all points being in a single cluster (divisive, much rarer).

In agglomerative hierarchical clustering, you first find two points (each in their own, single-point cluster at this point) that are closest to each other based on the distance matric (e.g. Euclidean distance, but others can be chosen depending on the domain of application). These two points become a new cluster. This step is repeated until all points are in a single cluster. At each step, we have a partitional clustering.

There is some nuance also in how the distance between two clusters is calculated. It can be done by finding the average position of the point-memmbers of a cluster or in other ways.

Nicely explained in [displayr](https://www.displayr.com/what-is-hierarchical-clustering/) and [wiki](https://en.wikipedia.org/wiki/Hierarchical_clustering).

Part of the reason hierarchical clustering beats K-means is that you don't have to specify the target number of clusters ahead of time. However, we always end up with a single cluster in the end (in the agglomerative version). So how do we use this to find the right number of clusters? 

Some answers are here on [vidyaanalytics](https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/). Essentially we obtain a dendrogram of the progressive clusters and choose a threshold distance value. This can be useful for e.g. Sales when they can have some flexibility as to how many different target groups (segments) of customers they can approach in different ways.

Finally, here's an actual [paper using supervised hierarchical clustering](http://proceedings.mlr.press/v97/yadav19a/yadav19a.pdf), from 2019. Notice the switch to using supervision.

### Autoregressive Models

Autoregressive means predicting the future behavior based on past behavior. In ML world it refers to seq2seq models that predict the next token based on previously predicted tokens (I believe).

### Graph Convolutional Networks (GCNs)

Very nice intro in [this article](https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b) by Inneke Mayachita. Inspired by Rylee Thompson's comments. The GCN's simplest version needs both an adjacency matrix $\mathcal{A}^{~n \times n}$ and a matrix of node features $\mathcal{R}^{~n \times d}$, which we then take the dot product of to get some new representation of the nodes as $\mathcal{H}^{~n \times d}$.

It's a generalization of the 2D convolution operation, looking at immediately (1-step away) connected neighbor nodes, in this simplest version.


### Skip-Connections 

Aka `Shortcut Connections` are a way to prevent very deep models from being hard to optimize. There are cases where a shallower model, with fewer layers, will perform better (even on the training set) than a deeper one, even if only identity layers were added. The problem of optimization can become harder for a deeper model.

Skip-connections solve this by either adding the output of an earlier layer directly to the output of later layers (skipping some layers) like in ResNet (residual connections) or by concatenating it, like in DenseNet. Whilst batch normalization and proper weights initialization by themselves should prevent the vanishing / exploding gradient problem, in practice these skip connections appear helpful too.

### Affinity Propagation

Is an adaptive clustering algorithm, a step above K-means, because it automatically chooses the optimal number of clusters (so it's adaptive). However, it still depends on so called "prototypes", i.e. it tries to learn what the prototype point for each cluster would be, and assigns points based on the distance from it. Thus it skews towards clusters in the shape of circles. This leads to incorrect assignments when the data consists of e.g. two crescents latching onto each other. An example of such a case can be seen [here](https://youtu.be/5O4aPDpRHpA?t=667). The step above this is DBSCAN, which solves for different cluster shapes (not just dispersions) - but DBSCAN doesn't guarantee that it will assign every element to a cluster (that depends on chosen parameters).

### Normalizing Flows

Normalizing flows are a representation learning technique, comparable to VAEs and GANs. In simple words, normalizing flows is a series of simple functions which are invertible, or the analytical inverse of the function can be calculated. For example, f(x) = x + 2 is a reversible function because for each input, a unique output exists and vice-versa whereas f(x) = x² is not a reversible function. Such functions are also known as bijective functions.

Good explanation [here](https://towardsdatascience.com/introduction-to-normalizing-flows-d002af262a4b). 

The normalizing flows transform a complex data point such as an MNIST Image to a simple Gaussian Distribution or vice-versa (and by distribution we mean it gets the $\mu$ and $\sigma$ of a $\mathcal{N}$ Normal distribution). Not clear to me how it's different from a VAE, but generally it is often presented side to side with GANs and VAEs that are capable of learning from unsupervised data.


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

### Ablation study

In AI, `ablation` is the removal of a component of an AI system, to see how the absence of that component impacts overall performance. In this way, we are able to somewhat judge its contribution to the overall results of the entire model. For neural nets this is an analogy to ablative brain surgery, where we tried to figure out what part of the brain does what by removing parts and asking animals to perform different tasks.

Ablative studies require that the system exhibits `graceful decomposition`, meaning that they continue to function (if worse) even with missing components.
