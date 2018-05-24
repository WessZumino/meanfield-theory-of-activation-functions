# Expectation propagation: a probabilistic view of Deep Feed Forward Networks
Code and data used for the numerical experiments reported in [arXiv:1805.08786](http://arxiv.org/abs/1805.08786)


We discuss a statistical mechanics model of deep feed forward neural networks (FFN). By recasting FFNs in terms of an energy based model one can consistently obtain several known results and unerstand the origin and limitations of some heuristics used in the deep learning literature. By providing a solid theoretical framework, we hope to provide new instruments that will help a systematic design of robust and explainable FFN.  

From a thereotical point of view, we infer that FFN can be understood as performing three basic steps: *encoding*, *representation validation* and *propagation*. We obtain a set of natural activations -- such as *sigmoid*, *tanh* and  *ReLu* -- together with a state-of-the-art one, recently obtained by Ramachandran et al. [Searching for Activation Functions](https://arxiv.org/abs/1710.05941), using an extensive search algorithm (there named *Swish*). 

We choose to term this activation *ESP* (Expected Signal Propagation), to stress its deep meaning and origin. In the main text we explain the probabilistic meaning of *ESP*, and study the eigenvalue spectrum of the associated Hessian on classification tasks. We find that *ESP* allows for faster training and more consistent performances over a wide range of network architectures.   

We provide two set of codes: a pure Python implementation, where the algorithm is described in detail, and a Tensor Flow (TF) implementation, that has been used to conduct the numerical analysis. The former is used mostly to show how backpropagation can be easily modified to be used with *ESP*, but the code is not particularly optimized. The algo is contained in the TF folder, in `esp_tf_utils.py` and commented in details. This needs to be loaded in the usual way in the file where the analysis is performed using 

                                       ` from esp_tf_utils.py import* ` 

In `analysis_utils.py` we include some useful functions for visualization and the function used to evaluate the eigenvalues of the Hessian matrix, together with the evaluation of the α and γ indices. The former measures the ratio of descent to ascent directions on the energy landscape; when α is large, gradient descent can quickly escape a critical point – a saddle point in this case – due to the existence of multiple unstable directions. However, when a critical point exhibits multiple near-zero eigenvalues, roughly captured by γ, the energy landscape in this neighbourhood consists of several near-flat (to second-order) directions; in this situation, gradient descent will slowly decrease the training loss, see paper for details. 

Finally, data and results are also provided, together with the videos showing the time evolution of the eigenvalue distribution of the Hessian matrix during learning. 
  
  





