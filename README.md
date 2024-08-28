# fuzzy-ml: Fuzzy Theory for Machine Learning in PyTorch :fire:
<a href="https://github.com/johnHostetter/fuzzy-ml/actions"><img alt="Actions Status" src="https://github.com/johnHostetter/fuzzy-ml/workflows/Test/badge.svg"></a>
<a href="https://github.com/johnHostetter/fuzzy-ml/actions"><img alt="Actions Status" src="https://github.com/johnHostetter/fuzzy-ml/workflows/Pylint/badge.svg"></a>
<a href="https://codecov.io/github/johnHostetter/fuzzy-ml"><img src="https://codecov.io/github/johnHostetter/fuzzy-ml/graph/badge.svg?token=WeWKlnVHqj"/></a>
<a href="https://github.com/psf/fuzzy-ml"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

The `fuzzy-ml` library is solely focused on utilizing fuzzy theory with machine learning, or 
implementing machine learning algorithms that derive fuzzy theory products (e.g., fuzzy sets, 
fuzzy logic rules). The library is designed to be used in conjunction with PyTorch and is built
on top of PyTorch's tensor operations, as well as the underlying `fuzzy-theory` library.

This separation of concerns allows for a more focused library that can be used in machine learning
applications, while still providing the benefits of fuzzy theory and fuzzy logic operations. The
`fuzzy-ml` library is designed to be easy to use and understand, with a simple API. Note that the
`fuzzy-ml` library is not intended to be used as a standalone library, but rather as a complement
to the `fuzzy-theory` library, nor is it intended to be used as a general-purpose fuzzy logic 
library, as it is specifically designed for machine learning applications.

## Special features :high_brightness:
1. *Quantitative Temporal Association Analysis*: Association analysis on temporal transactions with 
quantitative data is possible (e.g., Fuzzy Temporal Association Rule Mining).
2. *Clustering*: Fuzzy clustering algorithms are provided for unsupervised learning 
(e.g., Evolving Clustering Method, Empirical Fuzzy Sets).
3. *Partitioning*: Discover or create fuzzy sets to partition data for machine learning tasks
(e.g., Categorical Learning Induced Partitioning).
4. *Pruning*: Prune fuzzy logic rules to reduce complexity and improve interpretability 
(e.g., with rough set theory via rpy2).
5. *Rule Making*: Create fuzzy logic rules from data for interpretable machine learning models
(e.g., Wang-Mendel Method, Latent Lockstep Method).
6. *Summarization*: Summarize quantitative data to generate interpretable linguistic summaries.
