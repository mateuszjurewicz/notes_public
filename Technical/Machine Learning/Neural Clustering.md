# Neural Clustering

## Neural Clustering Process (2020)

Is a supervised clustering method that predicts an adaptive (learned) number of clusters, by Ari Pakman. It proposed two separate methods:

  1. Neural Clustering Process (pointwise) aka NCP
  2. Clusterwise Clustering Process (clusterwise) aka CCP

I will refer to pointwise NCP as just NCP. CCP, by contrast, predicts one cluster at a time. NCP is $O(N)$, CCP is $O(K)$, where $N$ is the cardinality of the input set and $K$ is the number of clusters. I have some doubts, from the implementation it looks like NCP is $O(NK)$, the paper claims to achieve $O(N)$ through parallelization, but I see it loop over all points and within that loop over all current clusters + one potential newly opened one.

Important to note that NCP/CCP does not **explicitly enforce permutation invariance** of the final output. So there's still a chance that it won't learn to make the same clustering choices with regards to two permutations of the same underlying set. Additionally, authors report a metric related to how permutations of the same set change the prediction.

NCP and CCP were tested on clustering mixtures of 2D Gaussians, clustering MNIST images into single-class clusters, and something called Spike Sorting, which is not an ordering task, but it's about grouping electrical neuron activity spikes as belonging to individual neurons.

It includes a form of variable-input length softmax for the prediction of a varying number of clusters, which just means there may be a different number of available clusters to assign a point to, but looking at code it's also not exactly a softmax function that in the end returns a vector over all available clusters, the highest being at the index of the chosen cluster.

### How does NCP work?

During training, NCP takes batches of points that are generated in such a way that a single target cluster assignment applies to every example in the batch, at least with 2D Mixtures of Gaussians. It then iterates over all available elements ($N$) and decides whether the current element should be assigned to one of the already _opened_ clusters or should become the first element of a new cluster. It is capable of predicting $K$ clusters, where $K \leq N$. 

During training, NCP is teacher-forced, in the sense that it uses the target (`cs`) to add the previously considered point ($n-1\textrm{th}$) to the internal representation of the cluster it actually belonged to. Target takes the form of an array referring to the arbitrary order of elements in the batched $X$ (`data`, in the code). For example:
```
# single-example batch of 3 points, in 2D (batch, N, elem_dim)
data = [[[0.9, 0.4], [0.3, 0.6], [0.1, 0.2]]]

# target 
cs = [0, 0, 1]

# also number of cluster members, padded at the start and end (doesn't seem to be used in the code)
clusters = [0 2 1 0]
```

In order to help the model learn to be **approximately invariant** to one of the 3 meaningless permutations (specifically of elements in $X$ aka `data`, since they all represent the same set) by default 6 different permutations of `cs` and `data` are obtained and essentially that means each batch becomes 6 batches during training. The other noisy, meaningless permutations are the order of points within clusters and the order of unassigned points.

### NCP Training
We will be using the following internal, learned representations:
1. `self.Hs` | previously opened (current) clusters, individually (b, K, 256)
1. `self.hs` | all elements in the set, individually (b, N, 256)
1. `self.Q` | all **remaining, unclustered elements** jointly (b, 256)
1. `self.qs` | all points in the set, individually (b, N, 256), used to represent **remaining, unclustered** elements
    - despite being $N$-sized, this is used to obtain `self.Q` by summing the remaining elements via indexing (`self.Q = self.qs[:, n,;].sum()`) during the very first interation, and then consecutively by subtracting the current points representation (`self.Q -= self.qs[:,n,]`)

There are also temporary representations that exist whilst looping over candidate clusters ($k \in K$) for the current point ($n^\textrm{th}$):
1. `Hs2` | previously opened (current) clusters, with the current candidate element ($n^\textrm{th}$) added to the currently considered ($k^\textrm{th}$) cluster.
    - `Hs2` is (b, k, 256)
1. `gs` | previously opened (current) clusters and the adjusted ($k^\textrm{th}$) cluster, individually (b, k, 512)
    - `gs` is (b, k, 512), it's a transformation of `Hs2` via `self.g()`. 
1. `Gk` | previously opened (current) clusters and the adjusted ($k^\textrm{th}$) cluster, jointly (b, 512), obtained from `gs` via summing.
1. `uu` | `Gk` and `self.Q`, concatenated (b, 256 + 512 = 768). Within the `k in range(K)` loop it represents all potential clusters and the remaining, unclustered points.

Finally there are 4 learned functions, which are all sequential stacks of `Linear()` and `PReLU`:
1. `self.h()` | obtains `self.hs` point representations from `data`, at the first element (`n==1`)
1. `self.q()` | obtains `self.qs` point representations from `data`, at the first element (`n==1`)
1. `self.g()` | obtains `gs` cluster representatiosn from flattened `Hs2`, at each `k` iteration plus another for the new potential cluster.
1. `self.f()` | obtains predictions from `uu` (the concatenated `Gk` and `self.Q`), at each `k` iteration plus another for the new potential cluster.

So, for each element $x_{n \in (0, N)}$ (`data[n]`) in a single example from the batch, during training, we proceed to:
  - pass the `data`, `cs` and current `n` to the `NeuralClustering()` model class.
  -  **(inference)** obtain the `logprobs` from the model, which are (batch, current k + 1), where current k is the number of clusters that we already assigned elements to (opened), plus 1 for starting a new cluster.
      - the prediction is teacher-forced, in that it checks at each n-th iteration whether the **previous** ($n-1\textrm{th}$) point should have been added to a new cluster or an existing one, and it either adds the representation of the previous point (from `self.hs`) to the correct cluster (via `self.Hs`) or concatenates the representation of the previous point (from `self.hs`) as the latest, unpredicted cluster (in `self.Hs`). 
      - in this way, when looping through current clusters the model has perfect awareness of what the correct assignments of preceding points were, and its internal representations of clusters (`self.Hs`) reflects that.
      - additionally, if the current element is **not** the last one, we update the joint representation of all the remaining elements (`self.Q`) by subtracting the representation of the current element from it (`self.Q -= self.qs[:, n, :]`).
          - if we are on the last element (the $n^{\textrm{th}}$), the representation of the remaining available points (`self.Q`) is set to all zeros.
      - the `logprobs` are now initiated to contain predictions per current K+1 clusters (making room for the potential newly opened one)
      - **(cluster loop)**: loop over all current clusters (`for k in range(K)`), **excluding the possibility of opening a new cluster** with current point as its only member:
          - initiate a new, temporary representation of current clusters (`Hs2 = self.Hs.clone()`).
          - add the current n-th point to the representation of the current candidate cluster (`Hs2[:,k,:] += self.hs[:,n,:]`)
          - then the clusters are flattened for the entire batch and embedded via the `self.g()` function into `gs`, which is a representation of all clusters **individually** (b, k, 512).
          - obtain the representation of all clusters `Gk` (b, 512), by summing over `gs`.
          - obtain the representation of all clusters and all available points `uu`, by concatenating `Gk` and `self.Q`. This joint representation of everything is now (b, 256 + 512 = 768).
          - make a per-cluster prediction about whether the current n-th point should be assigned to it by passing `uu` to `self.f()`, which outputs a single floating point number, which is then inserted into `logprobs` at the index `k` of the currently considered cluster (`logprobs[:,k] = torch.squeeze(self.f(uu))`)
      - **(potential new cluster)** is considered after the loop.
          - concatenate the representation of the current point (`self.hs[:,n,:]`) with `self.Hs` into the last `Hs2`
          - pass the flattened `Hs2` to `self.g()` to obtain `gs`
          - sum over `gs` to obtain `Gk` (b, 512), the representation of all clustered points (including the potentially clustered n-th point forming its own new cluster).
          - obtain `uu` by concatenating `Gk` with the representation of unassigned points `self.Q`
          - get the prediction regarding current point being in this new cluster by passing `uu` to `self.f()`, and put it in the last index of the tracked predictions for this point (`logprobs[:,K]`)
      - **(final logprobs)** are normalized into actual log values and somewhat normalized.
          - first, the highest predicted value is found (the predicted floating point number for the m-th cluster), marked as `m`.
          - every predicted value in logprobs (of length K+1) has `m` subtracted from it, resulting in a vector where every value is negative, except for the index of `m`, whose value becomes 0 (m - m = 0), the full operation is `logprobs = logprobs - m - torch.log(torch.exp(logprobs-m).sum(dim=1, keepdim=True))`
          - this results in a `logprobs` vector where the previously highest `m` value become the least negative one (so also most positive). Not clear to me what the purpose of this is.
  -  **(metric)** use `logprobs` and `cs` to see if the current element was assigned to the correct cluster, if so for the current example we track accuracy as 1.0, otherwise 0.0, but this is done for all n-th elements in the batch at the same time, so the actually accuracy per n-th element is a float between 0.0 and 1.0.
  -  **(loss)** is calculated by taking the predicted `logprob` of assigning the current n-th element to the **correct** cluster (based on the target) and subtracting it from the tracked loss value (initiated to zero), for every single element in batch, across the batch (mean reduced).
  - **(backwards)**, the loss is backpropagated only once all elements were assigned to clusters.

### NCP Sampling / Inference

Here's how the `NCP_Sampler()` class, which takes the instance of the `NeuralClustering()` model as `dpmm`, is used to make inferences on 1-elem batches. A new sampler is initialized for each 1-elem batch during the plotting function's execution.

During `NCP_Sampler()` initialization:

- `self.hs` is obtained via `model.h(data)`
- `self.qs` is obtainde via `model.q(data)`
- `self.f()` and `self.g()` are assigned to refer to `model.f()` and `model.g()` respectively.

During the call to `NCP_Sampler(S)`, the underlying model makes `S` predictions (`S` is th chosen number of samples), including their probabilities:

- (**before loop**): 
    - `cs` of size (`S`, `N`) is initiated. This will contain cluster assignments, sampled from the predicted probabilities, for all 5000 samples (`S = 5000`)
    - `previous_maxK` is set to 1
    - `nll` is initialized to all zeros of size (`S`). This will hold the actual probability of each sample.
- (**loop over $N$**) begins:
    - before checking if this is the first or later iteration:
        - `Ks` is initialized to `cs.max(dim=1)`. This holds the highest predicted cluster index for each sample.
              - so it tracks how many clusters are already opened in each sample. `Ks` is of size (`S`).
        - `Ks` gets 1 added to each entry, so that even from the start we have at least 1 cluster.
        - `maxK` and `minK` are obtained from `Ks`. These are integer values, tracking the maximum and minimum number of clusters already predicted
        in all samples so far.
        - `inds` dictionary is created, it's keys are all integers in range `(minK, maxK)` (inclusive), and the values are a Boolean vector over `Ks`. 
    - if **first iteration**:
        -  get `self.Q` by summing `self.qs[2:,:]` (excluding the first element). This represents remaining elements (jointly).
           - `self.Q` is (1, 256), `self.qs` is (N, 256)
        -  get `self.H` by inserting the first element of `self.hs` into it, at 0th index.
            - `self.hs` is (N, 256)
            - **but** `self.H` is initiated to be (`S`, 2, 256), where `S` is the number of chosen samples, and we already get representations of 2 clusters.
    - if **later iteration**:
        - first, we check whether `maxK` is larger than `previous_maxK`, and if it is:
            - we create `new_h`, which is all zeros (`S`, 1, 256), which represents an empty cluster.
            - we then concatenate `new_h` to the end of `self.H`
        - `self.Q` is updated to reflect the removal/subtraction of current point from remaining, unclustered points via `self.Q[0,:] -= self.qs[n,:]`
        or it's set to zeros if we're at the last element in the set.
        - `previous_maxK` is set to equal `maxK`.
    - `logprobs` are initiated to be (`S`, `maxK`) where `maxK` is the number of potential clusters to assign the current point to (2 even at first iteration, when the first point has to go into the first cluster).
    - `rQ` (`S`, 256) is initated from `self.Q`. It represents the remaining set of points (jointly) for all potential samples.
    - (**loop over $K$**) begins:
        - `K = maxK + 1` is used as loop range limit.
        - `Hs2` is obtained by cloning `self.Hs`. `Hs2` is (`S`, K, 256)
        - the current point's representation (`self.hs[n,:]`) is added into the currently considered cluster's representation `Hs2[:,k,:]`.
        - `gs` is obtained via `self.h()` to get a representation of all currently considered clusters (individually). `gs` is (`S`, `K`, 512).
        - (**big difference 1**) because each sample's prediction might have a different number of clusters already predicted, the `ind` is used
        to set the values of clusters that haven't been initiated yet in some samples to all 0s in those samples.
        - `Gk` is obtained by summing over `gs`. `Gk` is (`S`, 512) and represents all currently considered clusters, jointly as a set, for all samples.
        - `uu` is obtained by concatenating `Gk` and `rQ`, representing all currently considered clusters and remaining points. `uu` is (S, 768)
        - `logprobs` are obtained by passing `uu` to `self.f()`. 
            - `logprobs` contain the predictions of current $n^\textrm{th}$  point belonging to $k^\textrm{th}$ cluster, for all `S` samples. As the loop over $K$ continues, progressive indices in `logprobs` get filled with predictions for each possible cluster.
    - (**big difference 2**) because samples at this point may have a different number of clusters already predicted, the `inds` vector is used to set impossible predictions in `logprobs` to `-inf`. In this way we make sure the model isn't assigning the current $n^\textrm{th}$ point to a cluster beyond the potential newly opened one, I believe.
    - `logprobs` are normalized in the same way as in the train call.
    - `probs` are obtained from `logprobs` via exponentiation ($e^{x_{i}}$)
        - `probs` are (`S`, `K`)
    - `m` is initialized from `torch.distributions.Categorical(probs)`.
        - what this does is use the values from `probs` to return their indices, according to the probabilities belonging to them.
            - for example:
            ```
            m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
            m.sample()  # equal probability of 0, 1, 2, 3
            ```
    - `ss` is thus obtained by sampling, which gives us cluster assignments for current point, for all samples (`S`). `ss` is of size (`S`).
    - `cs` is constructed via `ss`, by placing the predicted/sampled cluster indices from `ss` into `cs` at $n^\textrm{th}$ index.
    - `nll` is all zeros at first iteration, and we keep subtracting from that zero the probability value we predicted for the target cluster, for each element. So `nll` starts with shape (`S`) and keeps that shape, and for each sample (each index) we keep subtracting the predicted probability of the target cluster, until we're out of elements.
- **(after N loop)**
    - `cs` is (`S`, `N`)
    - the `nll` is sorted and turned into probabilities, so that we know how probable each sample was.
    - the predicted cluster assignments are sorted from most probable to least, and returned.



### Questions / Open Problems:
1. CCP code is not available, it's not clear whether it will lend itself better to being extended to cluster ordering. Plan is to reach out to Ari Pakman.
1. **Extend NCP to Cluster Ordering** ideas:

     1. **NCP and Set2Seq training jointly on shared element embeddings** | we can use the same embedding of points for NCP and set-to-sequence, using the per-point order predictions from set2seq to then calculate the average order per cluster and treat that as a prediction.
        - we can possibly further improve it by using PMA as set pooling instead of summation!
        - still need to check papers that cite NCP (for ideas others have already implemented)

     1. **Simplest Set2Seq with repeated break marker** | we can use the simplest set-to-sequence approach with the section break token being always provided, and always available for pointing to it.
         - this can lead to model never finishing (always pointing to the page break and never having all elements selected)
     1. **CCP with pairwise predictions** | possibly with CCP, we could add a step that makes pairwise predictions regarding representations of already predicted clusters once
       a cluster is closed (but when do we know that it is closed?)


Sources:
- [github implementation](https://github.com/aripakman/neural_clustering_process) by the authors.

## Attentive Clustering Process (ACP) (2021)

ACP is a supervised neural clustering method capable of predicting an adaptive number of clusters. It offers an improvement over the Clusterwise Clustering Process (CCP) introduced in the NCP paper through the introduction of attention mechanisms. 

Pakman and Juho Lee found that **CCP underfits more complex datasets**, showcasing a need for greater expressive power, which ACP is supposed to bring. It was tested on graph community detection with adaptive (learned) number of communities, but the original code is also applied to a Mixture of Gaussians clustering problem much like NCP & CCP in this [github repo](https://github.com/aripakman/amortized_community_detection).

The paper introduces both ACP and ACP-S. ACP-S uses a simpler, anchor-independent aggregation with attention, explained in Appendix A (both methods). The anchor, I believe, is the uniformly sampled first point of the new cluster. Not sure how this changes things mechanically, it also seems ACP and ACP-S have similar performance.

### ACP training

1. model takes `data` and`labels`.
    - `data` is (b, N, 2) for Mixture of Gaussians (2 dimensional points)
    - `labels` is (N), presumably same trick with each example in batch having the same target. It takes the form of e.g. [0, 0, 1, 2, 2, 2, 3], in order.
2. within `self.forward_train()` we get cardinalities of each cluster as `cluster_counts`, in order, as a list.
    - e.g. if the 0th cluster had 10 elements and the 1st had 5, `cluster_counts` would be [10, 5]
3. `get_ATUS()` funtion gets called on the `cluster_counts`, returning 4 objects listed below. This function is an element of **teacher-forcing**. It gives the element indices that will be needed at each k-th step of ACP.
    - firstly, anchors get initialized to a list of as many 0s as there are clusters. E.g. [0, 0].
    - secondly, it obtains `cumsum`, which is a list of accumulated cluster cardinalities, preceded by a 0. In our example cumsum would be [0, 10, 15]
    - thirdly, we enter a **loop over  K**, the number of clusters in this example / batch (`k` in `K`)
        1. `all_anchors[k]` is set to `cumsum[k]`, identifiying the first point of each new cluster by its index in the original `data`, which is ordered ascendingly at this step.
        2. `all_targets` is a binary target vector over points available at `k`-th step, where points that should be assigned to `k`-th cluster get a 1, and the other available points get a 0.
        3. `all_unassigned` is the indices of unassigned points at each step `k`
        4. `last_assigned` is a list of the indices of all points in `k`-th cluster
4. `loss` and `elbo` get set to 0, these are the only output of `forward_train()` (no predictions are returned).
5. `MOG_Encoder` takes the data to embed it, this is just a stack of Linear() and PReLU(), ending on a Linear().
    - `enc_data` is returned and reshaped into (b, N, 128)
6. `sorted_ind` is obtained by passing `labels` (currently still sorted) to torch.argsort(), which returns the indices that would sort the values from labels. But because many labels have the same value (same cluster assignment), the output of argsort has an **element of randomness** to it.
    - e.g. in our example of [0 x 10, 1 x 5], the sorted_ind first 10 elements could have the indices of 0-10 in any order, e.g. [9, 2, 7, ...]. Only the elements after the 10th index would be guaranteed to be above 10, with the values between 10-15 in random over in that second sector.
7. `labels` are reordered according to the semi-shuffled, `sorted_ind`, not clear to me what this achieves.
8. `enc_data` gets reordered according to `sorted_ind` too, not clear what this achieves either, except the order of elements within clusters gets shuffled.
9. `G` is initialized to None.
10. **loop over k in K begins**, where `self.forward_k()` gets repeatedly called returning:
    - new loss term, new elbo term, new `G`
    - there's also a break condition if only 1 unassigned point remains to be clustered
    - `ind_anchor` is the index of the starting point (anchor) of current, `k`-th cluster, gotten from `all_anchors` at `[k]` index.
    - `ind_unassigned` is a list of indices of the points that are still unassigned, obtained from `all_unassinged` via `[k]` index.
    - `ind_last_assigned` is a list containing the indices of points from previous cluster  via `all_last_assigned][k-1]`, but **only if we're not on** `k == 0` (as there is no previous cluster then)
    - `targets`, a binary vector of 0s and 1s over the remaining available points is obtained from `all_targets` via `[k]` index.
    - from here, `self.forward_k()` gets called
11. `forward_k()` is called within the `K` loop.
    - it takes these as input:
        - `k` index of current cluster
        - `enc_data`, embedded elements
        - `G`, set to None at first cluster
        - `ind_anchor`, `ind_unassigned` and `ind_last_assigned`
        - `w`, the chosen number of VAE samples to generate (default 40)
        - `targets`, the binary vector of $y$ over unassigned, available points at k-th step.
    - it will return:
        - `loss_term` and `elbo_term` 
        - `G` for the next k+1 iteration.
        - the `loss` and `elbo` then get the terms added to them.

**ACP_Model.forward_k() call**

At this point we enter a big, main function that gets called at each iteration in `K`. Here's what happens inside it.

1. `self.update_global()` is called. 
    - it takes as arguments `enc_data`, `ind_last_assigned`, `G`, and `k`
    - it return a new `G`
    - if `k == 0`, `G` is set to all zeros in the shape of (b, 128)
    - else, we're not on the initial cluster:
        - if we're not on the very first cluster, `self.h()` is used to embed all individual representations of the points from the previous cluster (identified via `ind_last_assigned`) taken from `enc_data`. 
            - `self.h()` is a stack of Linear and PReLU, ending on Linear.
        - `hs_last_cluster` is the per-element representation of the last cluster
        - it gets passed to a PMA function and further to `self.g()`, resulting in a single vector representing the entire last cluster, of size (b, 128), which immediately gets added to `G` without forming an intermediate variable.
        - `G` is returned, updated via summation with the representation of the previous cluster.
2. `self.encode_unassigned()` is called.
    - its input: `enc_data`, `ind_anchor`, `ind_unassigned`
    - its output: `anch`, `data_unassigned`, `us_unassigned`
    - within, it checks if we're using attention, which I'll assume we are, this I believe is the difference between CCP and ACP (one parameter controls this).
    - `enc_all` is obtained by taking the representations of both the anchor point (via `ind_anchor`) and current candidate points (via `ind_unassigned`) through indexing into `enc_data`. 
        - `enc_all` is (b, N, 128) at first cluster, presumably N is replaced by the actual number of unassigned points at `k` plus 1 for the anchor point.
    - `self.ISAB1()` transforms `enc_all` into `HX`
        - `HX` is perm-equivar, same shape as `enc_all`.
    - `anch` is taken via indexing from `HX`
        - it's the new, attention-based representation of the anchor point.
        - it's (b, 128)
    - `us_unassigned` is also obtained by indexing into `HX`
        - takes everything from `HX` except the first element, which is `anch`
        - it's (b, n_available, 128)
    - `us_pma_input` is obtained via `self.MAB()`
        - this MAB takes `HX` and `anch` as input.
        - this is the **attention transformation that relates the anchor point to the available, unassigned elements**.
        - `us_pma_input` is (b, n_available + 1 for anchor, 128)
    - `U` is obtained via PMA from `us_pma_input`.
        - it is (b, 128)
    - `data_unassigned` is assigned to `us_unassigned`
    - `encode_unassigned()` returns `anch`, `data_unassigned`, `us_unassigned` and `U`.
    - I believe `U` is the encoding of all unassigned points in one vector, and `us_unassigned` / `data_unassigned` are a representation of all available, unassigned points individually. `anch` is the representation of just the current cluster's anchor point. Yes, and from the paper `G` is the encoding of already clustered points (into one vector, I believe).
3. `self.get_pz()` is called.
    - its input: `anch`, `U`, `G`
    - its ouput: `pz_mu`, `pz_log_sigma`
        - **mu and sigma are parameter vectors controlling the distribution of the k-th cluster, specifically the mean and covariance of the Gaussian Mixture Component** (paper page 2, blue comment)
    - `self.pz_mu_log_sigma()` is called on a **concatenation of `anchor`, `U` and `G`**.
        - the input concatenated vector is (b, 128*3)
        - this is a neural net function consisting of a stack of Linear and PReLU, with a final Linear layer.
        - it returns `mu_logstd`, which is (b, 256)
        - I think this is meant as a representation of both assigned (`G`) and unassigned points (`U`), as well as the anchor.
    - `mu` and `log_sigma` (mean and log standard deviation) are obtained from `mu_logstd` by cutting it in half at z_dim = 128th index
        - `mu` is (b, 128), first half of `mu_logstd`
        - `log_sigma` is (b, 128), second half of `mu_logstd` via mu_logstd[:, self.z_dim:]
        - both will only be used in the ca;culation of the loss (by then it's called `pz_mu`)
4. Conditionally, if `targets` is None, we go into a short function that exists `forward_k()` by returning logits via `self.vae_likelihood()`
5. `self.conditional_posterior()` is used to obtain `qz_mu` and `qz_log_sigma`.
    - its input: `us_unassigned`, `G`, `anch`, `targets`, `w`
    - first, it obtains boolean as opposed to binary target vector, `t_in` from `targets`
    - then it figures out the `reduced_shape` by taking the zeroth and second shape from `us_unassigned.shape`)
        - at first `k`, `reduced_shape` is (b, 128)
    - checks if `t_in` is all False
        - if it is, sets `U_in` to an all zeros vector of `reduced_shape` size.
        - else it obtain `U_in` via `self.pma_u_in()`
            - `self.pma_u_in()` takes as input `us_unassigned[:, t_in, :]`, so it is also teacher-forced (boolean indexing!).
            - it is a PMA module.
            - it returns `U_in`, which is (b, 128) and I think it's a single vector representing all unassigned points.
    - check if `t_in` is all True (a situation where all available points should be in current cluster according to the target?)
        - if it is, `U_out` is set to all zeros, according to `reduced_shape` size.
        - else it obtains `U_out` via `self.pma_u_out()`
            - `self.pma_u_out()` takes as input `us_unassigned[:, ~t_in, :]`. Notice the negation (`~`) boolean indexing.
            - I believe `U_in` represents the unassigned points that are meant to be in the current cluster in the target (as 1 vector) and that `U_out` represents the ones that are out-of-kth-cluster.
        - both `U_in` and `U_out` are (b, 128)
    - `self.get_qz()` is used to obtain `qz_mu` and `qz_log_sigma`
        - its input is `anch`, `U_in`, `U_out` and `G`.
            - these are the representations of the current (k-th) cluster's anchor point, the points that are meant to be in this cluster, the points meant to be out of this cluster and the already previously clustered points (`G`), each as a single vector, each of size (b, 128).
        - inside `self.get_qz()`, another function is called much like with `self.get_pz()` we called `self.get_pz_mu_log_sigma()`.
            - `self.qz_mu_log_sigma()` is called, taking a single concatenation of `anchor`, `U_in`, `U_out` and `G` (size b, 128*4=512)
                - it s a stack of Linear and PReLU, ending on Linear
            - it returns `mu_logstd`, which is indexed into at `z_dim`, splitting itself into two halves:
                - `qz_mu` (`qz` version) is (b, 128)
                - `qz_log_sigma` (`qz` version) is (b, 128)
        - `torch.distributions` is used as `dist` to sample from the Normal distribution
            - it is parameterized by the mean `qz_mu` and the exponentiatied `qz_log_sigma.exp()`
            - this distribution is assigned to `qz_b`
        - `qz_b` distribution is sampled from via `.rsample()`
            - `rsample()` allows for pathwise derivatives, as opposed to sample(), relating to the reparametrization trick.
            - `z` is thus obtained by this sampling
    - `z` is (`w`, b, 128)
    - `self.conditional_posterior()` returns:
        1. `qz_mu` (b, 128)
        2. `qz_log_sigma` (b, 128)
        3. `z` (w, b, 128), so 40 (by default w=40) VAE samples.
6. `self.vae_likelihood()` is called.
    - its inputs: `z`, `U`, `G` , `anch`, `data_unassigned`
    - its outputs: `logits`
    - number of currently unassigned points (`Lk`) is extracted from `data_unassigned`, which is a per-elem embedding of unassigned elements.
    - `expand_shape` is set to (-1, `Lk`, `w`, -1)
    - `zz`, `dd`, `aa`, `UU`, `GG` get obtained from `z`, `data_unassigned`, `anch`, `U` and `G` respectively via reshaping to `expand_shape` dimensions.
    - `ddzz` is obtained through concatenating the 5 new vectors above.
        - `ddzz` is (b * `Lk` * `w`, 128 * 5 = 640)
    - `logits` are obtained by calling `self.phi()` on `ddzz`.
    - `self.phi()` is a stack of Linear and PReLU, with last Linear.
    - `logits` is (b, num unassigned points, `w`), returned here.
        - I think these are batched probabilities per each unassigned point, for each of the default 40 (`w`) VAE samples.
7. `self.kl_loss()` is called to get loss and elbo.
    - its inputs: `qz_mu`, `qz_log_sigma`, `pz_mu`, `pz_log_sigma`, `z`, `logits`, `targets`.
    - its output: loss and elbo
    - at this point `targets` is a 1-dim vector of size equal to the number of unassigned elements, matching `logits` along their middle dimension.
    - within `self.kl_loss()`:
        - a Bernoulli distribution object is instantiated based on the `logits`.
        - `targets` get expanded to batch size and number of VAE samples (`w`)
        - quite a bit of other operations happen using the `qz` and `pz` mean and stdev, and some clever juggling with no gradients.
8. `forward_k()` returns loss, elbo, `G` for next cluster and `logits`, but that last object is ignored within `forward_train` (set to _)
9. next cluster's loop within `forward_train()` begins, with an updated `G` representing previously clustered elements.

**ACP_Model inference**

Following `acp_cluster_smb.py`, from a graph-based model.

1. `SelectedSampler` class instance is created based on reloaded `model` and `data`
    - `data` is a single example from `data_generator` via its `generate_single()` method, of size (1, N, N).
    - it is the `ACP_Sampler()` class instance
2. Within the `ACP_Sampler`:
    - `self.enc_data` variable is set to be the result of the model's `model.encoder(data)` call.
        - in the graph setting, this `model.encoder` is a GraphSageEncoder.
        - `self.enc_data` is now (N, 128)
    - `self.hs` is obtained by calling `model.h()` on the `enc_data`.
        - `model.h()` is a sequence of Linear and PReLU
        - `self.hs` is now (N, 128)
        - however the two 128s above come from different params (e_dim and h_dim), set to the same value.
    - if we weren't using attention, we would also obtain `self.us` here, from `enc_data`, via `model.u()`, but we don't.
3. At this stage, we have an instantiated `SelectedSampler`, which is an `ACP_Sampler` with `self.enc_data` and `self.hs` corresponding to the single example of data used during its instantiation.
    - **inference is per single example!**
4. The sampler's `sample()` function is used.
    - it outputs 
        - `clusters`: (1, N)
            - this is just a vector of class labels over N
        - `probs`: (1)
    - its inputs: 
        - `args.S`, presumably the number of samples to generate, set to 10
        - two variables set to False, called sample_Z and sample_B
        - `args.prob_nZ`, set to 1
        - `args.prob_nA`, set to 10
5. `sample()` includes:
    1. a call to `self._sample()` which only receives `S` equal to 10.
        - returning `ccs`
    2. followed by a call to `self._estimate_probs()` which takes `ccs` and returns `cs` and `probs`, which are the final output of the wrapping `sample()` call.
6. `self._sample()` includes:
    - obtainin current `N` from `self.enc_data`
    - `G` is initialized to all zeros of size (`S`=10, 128)
    - `cs` is initialized to be all -1s of size (`S`, `N`)
    - if attention is used, we get `big_enc_data` from viewing `self.enc_data` into the shape of (`S`, `N`, 128)
    - `big_hs` is obtained by viewing `self.hs` into size (`S`, `N`, 128)
    - `mask` is initialized as all 1s of size (`S`, `N`), it will keep track of unassigned indices of elements in each of the `S` threads
    - **loop begins** while `t` > 0, where `t` is initially equal to `S` (10), and it keeps track of how many threads have not completed their sampling yet (i.e. have available elements to cluster).
        - `k` is set to -1 before the loop
        - `k` is incremented by 1 to 0, first thing in the loop.
        - `anch` is created from a call to `torch.multinomial()`, returning a distribution over some number of classes, adding to 1. In this case, the number of classes depends on the `mask[:t, :]`. 
            - `anch` is of size (10, 1) at first.
            - it contains 10 integers within the range 0 to `N`
        - label `k` is assigned to the anchor elements
            - this is done via `cs` and a call of the tensor `self.scatter_()` method, which writes the values from a given tensor, into the source (self) tensor, at given indices.
            - in our case, `cs`'s values (originally all -1s) at the indices specified by the `anch`, is set to `k`. This is only done up to `t` dimension, only affecting still open threads (`cs[:t, :].scatter_(1, anchs, k)`).
            - **the effect** of this is that `cs` now has a single 0 in each of its 10 vectors of length `N`. So we randomly choose a single (but different) element in each row of `cs` to be in the first, `k=0`th cluster.
        - if we use attention:
            - `HX` is obtained via `self.model.ISAB1` taking as input `big_enc_data` and `mask`, both indexed into via `t`.
                - `HX` is (`S`, `N`, 128)
            - `A` is obtained from `HX` by indexing into it using `t` and `anchs`. This is just a single vector that contains the embedded representations of each of the anchor points, whose indices are in `anchs`.
                - `A` is (`S`, 128)
            - `us_pma_input` is obtained via `self.model.MAB()` called on `HX` and `A`.
                - `us_pma_input` is (`S`, `N`, 128), representing individual unassigned elements? But not excluding anchor points? Shouldn't this be `N-1`?
            - `U` is obtained via `self.model.pma_u()` called on `us_pma_input` and `mask[:t, :]`. At the start this just means the entire `mask`.
                - `U` is (`S`, 128), a single vector representing all unassigned points jointly.
            - **selected anchors are removed from the mask**
                - this is done by changing the value of elements in `mask` from 1 to 0 at the indices from the `anchs`, but only in still open threads, via `[:t, :]`.
            - `Dr` is set to equal `HX`
                - `Dr` is (`S`, `N`, 128), an encoding of all elements, stemming from `big_enc_data`.
        - unless sample_Z is set to True, we use `self.model.get_pz()` called on `A` `U` and `G[:t, :]` to obtain `Z`.
            - so we call this learned neural function on the single-vec representations of all anchors, all unassigned elements and all assigned elements, getting encoding `Z` of size (`S`, 128). 
                - `model.get_pz()` calls `model.pz_mu_log_sigma()` under the hood, which is a stack of Linear and PReLU, returning mu and log_sigma, from a single vector split in half.
                - `Z` is mu, log_sigma is discarded (set to _)
            - `Z` is unsqueezed to (1, `S`, 128)
        - `Ur` is obtained from `U` via viewing and expanding, to size (`S`, `N`, 128)
        - `Ar` is obtained from `A` via viewing and expanding, to size (`S`, `N`, 128)
        - `Zr` is obtained from `Z` via viewing and expanding, to size (`S`, `N`, 128)
        - `Gr` is obtained from `G` for current `t` open threads, size (`S`, `N`, 128)
        - `phi_arg` is obained by concatenating `Dr`, `Zr`, `Ar`, `Ur` and `Gr` and reshaping it into size (`t` * `N` , 128 * 5 = 640)
        - `logits` are obtained from `phi_arg` via `self.model.phi()`, which is a stack of Linear and PReLU ending on a binary Linear (like regression), predicting a single float.
            - `logits` are size (`t`, `N`)
        - `prob_one` is obtained from the `logits` (which are real valued) into probabilities (0-1), via exponentiation and some more juggling.
        - **this is big**: it seems there's a threshold of above 0.5 for `prob_one`, above which `inds` are set, which are a Boolean vector of size (`t`, `N`).
            - unless sample_b is set to True, then it's more randomly sampled according to the probability in `prob_one`.
            - it appears that `inds` can sometimes be all False, not sure if that's because the model is almost untrained here or if this ACP method allows unassigned... I think it's actually just that the single anchor point will be in this cluster.
        - `sampled` is obtained by castin `inds` into Long type.
        - `sampled_new` is obtained by multiplying the `mask` (but only indexed into via current `t` number of threads) with the `sampled`.
            - at this point `mask` is all 1s with a single 0 at the index of the anchor point
            - `sampled` is all 0s except for where the other chosen elements for this cluster would be.
            - so it seems the model continues to predict even for points that were already assigned, but the mask prevents them from being actually sampled (put into?) the current cluster.
        - `new_points` are initialized as a flat vector unpacking `sampled_new` into a size of (`t` * `N`). These are flattened indices of new points for cluster k. E.g. [667, 671 ...1671], because it's t times N, which is 10 * 222 at first.
        - `cs` is updated to reflect these assignments.
            - `cs` elements at `new_points` indices are set to equal `k`, the label of the current cluster.
            - `cs` is size (`t`, `N`)
        - `mask_update` is created as 1 - `sampled`. 
            - it has 1 on the points that survived the last samping.
        - `mask` is update with `mask_update` via multiplicaton, but only for the remaining `t` threads / samples.
        - `new_cluster` is created from `cs` where its elements equal `k`. It seems like a binary vector.
        - using attention, `new_Hs` is obtained via `self.model.pma_h()` from `big_hs[:t]` and the `new_cluster`.
            - `new_Hs` is size (`t`, 128)
        - the representation of previously completed clusters is updated with the `new_Hs` via:
            - `self,model.g()` is used to transform `new_Hs` which is immediately added to `G[:t]`.
        - `msum` is obtained by summing over `mask[:t]`, along the first dimension (`t`).
            - `msum` is a tensor with `t` integer values, which represent the number of remaining unassigned elements for each sample / thread.
        - `msum` is used to see if any of the samples / threads can already be closed (if any of the `t` values in `msum` is 0).
        - if any thread can be closed:
            - `msumfull` is obtained from `mask.sum()`, resulting in a vector of integers of size (`t`). I believe this shows which threads are closed, marked by 0 of available points to assign.
            - `mask`, `cs`, `G` and `t` are adjusted based on how many remaining threads there are.
            - **this is important** because `t` being adjusted and subtracted from is an end condition of the while loop!
    - **loop over t ends**
    - `cs` is relabelled, such that cluster labels are in order of appearance, according to the arbitrary order of points. E.g. from [3, 2, 1, 2] we'd go to [1, 2, 3, 2].
    - this is done in a loop for each sample in `S`.
        - **I think this might be what prevents unassignment**, since -1 placeholder values will get turned to sth else.
    - duplicates are eliminated from `cs`, as it gets turned into a set() and into `lcs`
        - `lcs` is a list of up to `S` tuples of `N` elements, containing cluster labels for each thread / sample.
    - `Ss` is set to the length of `lcs`
    - `ccs` is set to all zeros of size (`Ss`, `N`)
    - `ccs` is fed from `lcs`, to take all of its values, ending up as of size (`Ss`, `N`)
    - `ccs` is returned.
7. We are back ub the call to `sample()` (within which `_sample()` was called to obtain `ccs`)
8. `self._estimate_probs()` is called
    - its inputs: 
        - `ccs`, a numpy array of size (number of unique samples, `N`)
        - `prob_nZ` = 1
        - `prob_nA` = 10
    - its ouptuts:
        - `cs` and `probs`, the final outputs of the whole sampler. 
9. within the call to `_estimate_probs()`:
    - `S` is obtained from `cs`, it's the number of samples that were unique (10 in this case)
    - `N` is the number of elements in the single example.
    - `probs` are initialized to a numpy array of all 1s of size `S`
    - **loop over s in S begins**
        - `K` is set to the max value from the `s`-th sample in `cs` + 1. That's the total number of clusters predicted in that thread / sample (since cluster labels are zero-indexed).
        - `G` is initialized to all 0s of size (128)
        - `Ik` is arange() into array of size `N`, it's the available indices before sampling cluster k.
        - **inner loop over k in K begins**
            - `Sk` is set to `cs[s, :]] == k`, it holds all points that could be possible anchors for `k`-th cluster.
            - `nk` is set to `len(Ik)`, which at this point equal `N`.
            - `ind_in` and `ind_out` are vectors of indices of points in and out of current cluster according to `ccs` respectively.
            - if `nA` is None or if the `Sk.sum() < nA`:
                - `sk` is set to `Sk.sum()`
                - `anchors` are set to `ind_in`
            - else:
                - `sk` is set to `nA`
                - `anchors` are equal to randomly chosen `sk` number of elements from `ind_in`.
            - since in our examples `nA` is set to 10, we only go into the if condition when the number of potential anchors is smaller than 10.
            - `d1` is equal to arange of len `sk`. So [0, 1, 2 ... 9]
            - if attention is used, we get `HX` from `self.model.ISAB1()`, which gets called on `self.enc_data[Ik, :]`.
            - `bigHx` is obtained from `HX` via viewing and expanding
            - `A` is obtained from `HX` via indexing through `[anchors, :]`
            - `us_pma_input` is obtained via `self.model.MAB()` from `bigHx`
            - `U` is obtained via `self.model.pma_u()` from `us_pma_input`
            - `Dr` is set to `HX`
            - `Ge` is obtained from `G` via viewing and expanding
            - we get `Z` via `self._sample_Z()` from `A`, `U`, `Ge`, `nZ`
            - we get `Ar`, `Dr`, `Ur`, `Gr`, `Zr` from their respective variables without the r, via viewing and expanding.
            - `phi_arg` is obtained by concatenating `Dr`, `Zr`, `Ar`, `Ur` and `Gr`.
            - `logits` and obtained via `self.model.phi()` from `phi_arg`.
                - `logits` are also viewed to shape (`nZ`, `nk` , `sk`)
            - `prob_one` is obtained from logits via spicy exponentiation (1 / (1 + torch.exp(-logits)))
            - `prob_one[:, anchors, d1]` is set to 1. I think this is setting probs of anchor points belonging to current cluster to 1.
            - `pp` is initialized to be equal to `prob_one[:, ind_in, :].prod(1)`. The prod(1) function on the tensor returns the product of its elements along the 1st axis.
                - more operations are done on `pp`
            - preparations for next iteration:
                - `Hs` is created from `self.hs[Sk, :]`
                - it is then transformed into itself by `self.model.pma_h(Hs)`
                - `G` is += `self.model.g(Hs)`
                - available indices are update (`Ik`)
        - `inds` is set to indices of predictions in decreasing order of probability
        - `probs` = `probs[inds]`
        - `cs` is sorted according to the `inds`
        - `cs` and `probs` are returned.
            - `cs` is (`S` - nonunique, `N`), it's the final cluster labels.
10. **ccs and cs are NOT the same** but seem close. Approximately 10-50% of elements got their cluster labels changed in a couple investigated examples. Sometimes the cluster label is different by 1, sometimes by up to 3.
11. Outside of the sampler, `predicted` is set to `cs[np.argmax(probs)]`. 

### VAE element in ACP

The mean and std dev are obtained from a stack of linear and prelu, split in the half, so that they're both e.g. 128 dimensions. Both `pz` and `qz` are obtained this way from the representations of assigned and unassigned points, as well as the anchors. The `z` is sampled from `qz`, to be some actual representation. `z` is then used in the `vae_likelihood()` function to obtain the logits, in that it is concatenated to all the other representations and passed to that final `phi` function that is also a stack of linear and prelu, outputting the logits.

In the `kl_loss()` the logits are compared to the targets, by first obtaining a Bernoulli distribution ampler from them (given their predicted probabilities), giving us `lpz_b`. The `qz` is used to instantiate a Normal distribution sampler, and we get the probability of `z` from this sampler, as `lqz_b`. Same way we get `lpz` from `pz` and `z`. 

The loss and elbo are ultimately obtained from `lw` which is equal to `lpz + lpb_z - lqz_b`. This makes sense because depending on how likely the sampled `z` was based on `qz`, we will want to adjust how much we reward or punish the model.


### Open Questions:
- How does ACP guarantee that every element will be assigned to a cluster if it uses a threshold value of 0.5? (during inference)
    - **Answer**: it has a while loop over `t` = `S`, which is the chosen number of samples. By default it's 10. At the start of every iteration, we choose 1 anchor point. So at worst, this loop will have `N` iterations. It is entirely possible for the neural functions to result in logits that result in per-element probabilities that don't go over the **0.5 threshold**, which would mean that no *other* element (other than the anchor) gets added to the current cluster (one cluster per while loop iteration). However, the anchor point still gets added and thus removed from the pool of available elements / points. So it can't get stuck in a forever while loop, nor can it result in unassigned elements (since one point will get added).
        - it remains unclear how this would get solved in the ACP-S version that doesn't use anchor points. There's `acps_model` in the `models` repo of the amortized community detection project!
        

Sources:
- [Ari Pakman's github repo for ACP](https://github.com/aripakman/amortized_community_detection)