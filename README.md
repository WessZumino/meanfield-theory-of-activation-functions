# Expectation propagation: a probabilistic view of Deep Feed Forward Networks
Code and data used for the numerical experiemnts reported in arXiv:


We discuss a statistical mechanics model of deep feed forward neural networks (FFN). By recasting FFNs in terms of an energy based model one can consistently obtain several known results and unerstand the origin and limitations of several heuristics used in the field. By providing a solid theoretical framework, we hope to provide new instruments that will help a systematic design of robust and explainable FFN.  

From a thereotical point of view, we infer that FFN can be understood as performing three basic steps: *encoding*, *representation validation* and *propagation*. We obtain a set of natural activations -- such as *sigmoid*, *$\tanh$* and  *ReLu* -- together with a state-of-the-art one, recently obtained by Ramachandran et al. [Searching for Activation Functions](https://arxiv.org/abs/1710.05941), using an extensive search algorithm (there named Swish). 

We choose to term this activation *ESP* (Expected Signal Propagation), to stress its deep meaning and origin. In the main text we explain the probabilistic meaning of *ESP*, and study the eigenvalue spectrum of the associated Hessian on classification tasks. We find that *ESP* allows for faster training and more consistent performances over a wide range of network architectures.   

We provide two set of codes: a pure Python implementation, where the algorithm is described in detail, and a Tensor Flow implementation, that has been used to conduct the numerical analysis. The former is used mostly to show how backpropagation can be easily modified to be used with *ESP*, but the code is not particularly optimized. The algo is contained in `esp_tf_utils.py` and commented in details. This needs to be loaded in the usual way in the file where the analysis is performed using 

                                       `from esp_tf_utils.py import*` 

In `analysis_utils.py` we include some useful functions for visualization and the function used to evaluate the eigenvalues of the Hessian matrix, together with the evaluation of the $\alpha$ and $\gamma$ indices, see paper for description. 
Finally, data and results are also provided, together with the videos showing the time evolution of the eigenvalue distribution of the Hessian matrix during learning. Below we provide a short account of the main results of our work. 



