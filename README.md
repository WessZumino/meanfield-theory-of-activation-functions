# Expectation propagation: a probabilistic view of Deep Feed Forward Networks
Code and data used for the numerical experiments reported in arXiv:


We discuss a statistical mechanics model of deep feed forward neural networks (FFN). By recasting FFNs in terms of an energy based model one can consistently obtain several known results and unerstand the origin and limitations of some heuristics used in the deep learning literature. By providing a solid theoretical framework, we hope to provide new instruments that will help a systematic design of robust and explainable FFN.  

From a thereotical point of view, we infer that FFN can be understood as performing three basic steps: *encoding*, *representation validation* and *propagation*. We obtain a set of natural activations -- such as *sigmoid*, *tanh* and  *ReLu* -- together with a state-of-the-art one, recently obtained by Ramachandran et al. [Searching for Activation Functions](https://arxiv.org/abs/1710.05941), using an extensive search algorithm (there named *Swish*). 

We choose to term this activation *ESP* (Expected Signal Propagation), to stress its deep meaning and origin. In the main text we explain the probabilistic meaning of *ESP*, and study the eigenvalue spectrum of the associated Hessian on classification tasks. We find that *ESP* allows for faster training and more consistent performances over a wide range of network architectures.   

We provide two set of codes: a pure Python implementation, where the algorithm is described in detail, and a Tensor Flow (TF) implementation, that has been used to conduct the numerical analysis. The former is used mostly to show how backpropagation can be easily modified to be used with *ESP*, but the code is not particularly optimized. The algo is contained in the TF folder, in `esp_tf_utils.py` and commented in details. This needs to be loaded in the usual way in the file where the analysis is performed using 

                                       ` from esp_tf_utils.py import* ` 

In `analysis_utils.py` we include some useful functions for visualization and the function used to evaluate the eigenvalues of the Hessian matrix, together with the evaluation of the α and γ indices. The former measures the ratio of descent to ascent directions on the energy landscape; when α is large, gradient descent can quickly escape a critical point – a saddle point in this case – due to the existence of multiple unstable directions. However, when a critical point exhibits multiple near-zero eigenvalues, roughly captured by γ, the energy landscape in this neighbourhood consists of several near-flat (to second-order) directions; in this situation, gradient descent will slowly decrease the training loss, see paper for details. 

Finally, data and results are also provided, together with the videos showing the time evolution of the eigenvalue distribution of the Hessian matrix during learning. Below we provide a summary of the main ideas and results of our paper. 

## Expected Signal Propagation 

Consider the original, biologically inspired, representation of a single computational unit (neuron) as performing three steps: 

1. **encoding**: The original information is encoded to a lower/higher dimensional representation. The former case corresponds to compression, or coarse graining in physics: the new, coarse grained, variable  - h - contains information that is *relevant* (more einformative) on the new lenght scale we are intered in. For example, the single pixel of an image are not very informative to determine how the image looks like; only by coarse graining (zooming out) we can relate the microscopic values of the pixels to the macroscopic image. In Physics, coarse graining allows one to understand how microscopic constituents self-organize in order to give rise to macroscopic properties of matter. In both example, at each step of coarse graining one determines the relevant, emerging features in the system. 

2. **representation validation**: once the new representation has been created, it needs to be validated. In the neuron this step is accomplished via comparison with a local bias potential that, in turns, determines the probability of the gate (the synaptic button) to be open or close. So, if h is an informative representation of the input information, given the output, then the gate s=1. The expectation value <s> for such a binary state is the sigmoid function if s=(0,1) or tanh if s=(-1,1). 
  
3. **propagation**: if the gate is open, then the neuron should transmit the expectation value <h> to the other neurons in the next layer. This simple step is missing in traditional account of Deep Learning (DL). That this is the right quantity to transmit is easy to understand if one regards the system as composed of physical neurons or as a circuit: <s> is related to the probability that the gate is open or not, but it is not the result of the computation. If one transmits this quantity, as in traditional DL, it is easy to see that a dimensional mismatch occurs. Let us measure the incoming information (the external data)in bits; then, in the first layer h_1 has dimensions of bits, being a linear combinaition of the incoming input. This means that the activation a_1= <s_1> = sigma(beta h_1) is dimensionless (being a distribution) and beta has therefore dimension of 1/bits. The constant beta is the inverse noise parameter and it is arbitrary fixed to 1 in DL. Let us move now to the next layer, the DL recipe is to take a linear combination of <s_1>, leading to the coarse grained variable h_2 to be dimensionless, and thereore a_2 = <s_2> = sigma(beta h_2) is now dimensionful!     
This does not happen if one transmits instead <h>, as it can be easily checked. For beta --> infinity (noiseless limit) this expectation value returns the ReLu function. 
  
A schematic representation of the single unit behaviour is ![given below](https://github.com/WessZumino/expectation_propagation/tree/master/Images/neuron.pdf)

  
  
  
  
  
  
  





