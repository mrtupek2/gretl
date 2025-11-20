# Gretl
Graph-based reversible evaluation and tangents library

GRETL is a C++ library for evaluation, re-evaluation and algorithmic differentiation of functional
operations on an arbitrary computational graph with limited memory usage. Similar to popular
machine learning frameworks in python, like pytorch and jax, it tracks and stores both operations
and output data as functions are evaluated. Once this composition of functions is built up, the
entire chain of operations can be backpropagated to compute sensitivities of the final result with
respect to any number of inputs. An important distinguishing feature of GRETL is its ability to
limit the maximum memory usage by automatically dynamic checkpointing the data output for
each graph operations (see Wang, Moin, Iaccarino, 2009). During backpropagation, parts of the
graph that are no-longer in memory are automatically re-evaluated from upstream checkpointed
states as needed for derivative sensitivity calculations (more precisely, vector-Jacobian products).

In contrast to most machine learning applications, memory usage becomes an essential bottleneck
for many physics applications, especially time-integrated solutions to PDEs where dynamic checkpointing 
becomes essential. GRETL is particularly beneficial for applications, such as coupled
multi-physics, where deriving adjoint-based sensitivities and managing checkpoint memory across
modules becomes onerous. It is meant
to be used at a fairly high level, as the graph construction and memory management entails some
overhead. A good rule of thumb for scientific applications is GRETL operations should be around
parallel communication calls, not within them. Cases which can be readily handled by the GRETL
library include: different time-integration algorithms per physics (e.g., coupled predictor-corrector
algorithms, IMEX, etc.), sub-cycling, asynchronous integrators, state-dependent stable-timesteps,
iterative solvers, coupling algorithms, controller algorithms, and more.

License
-------

Unlimited Open Source - BSD 3-clause Distribution
See [LICENSE](./LICENSE) for details.

`LLNL-CODE-2013480`
