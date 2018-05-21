"""
Helper functions for FNN with ESP
=================================================================
Author: Mirco Milletari <mirco@bambu.life>

This is a pure python implementation of a Feed Forward Deep network with ESP
activation. See main text for details.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
#Math Libraries
import numpy as np

#Visualization libraries
import matplotlib.pyplot as plt

#metrics
import sklearn
import sklearn.datasets
from sklearn.metrics import r2_score

np.random.seed(1)

#=====================
# DNN Initialization
#=====================

#Sigmoid Function, aka the Fermi-Dirac distribution
def sigmoid(beta, h):
    s= 1/(1+ np.exp(-beta*h))
    return s

#Parameter initialization
def initialize_parameters(layers):

    '''
    Initialise the parameters of the model:

    L-- number of layers in the network (excluding the ouput)
    W-- weight matrix, dim: (l, l-1) initialized to a small number drawn from a standard normal distribution
        mean 0 and std 1.
        Different initializations:  *np.sqrt(2/layers[l-1]), sum_j omega_{i,j}=1
        The second initialization makes sure that the sum of the weights is 1 for each column

    b-- bias vector, dim: (l,1)
    beta-- inverse "temperature". initialized according to randn, but beta>0. We Initialise beta small, i.e.
           high temperature. Note that each unit has its own beta as it attains only local equilibrium.
           Another possible initialization of beta is to 1 for each unit.
    '''
    np.random.seed(1)

    parameters = {}
    L = len(layers)            # number of layers in the network

    for l in range(1, L):

        parameters['w' + str(l)] = np.random.randn(layers[l], layers[l-1])
        norm= np.sum(parameters['w' + str(l)],1).reshape(layers[l],1)
        parameters['w' + str(l)] = parameters['w' + str(l)]/norm

        parameters['b' + str(l)] = np.zeros((layers[l], 1))
        parameters['beta'+str(l)]= np.random.randn(layers[l],1)*0.1

    return parameters


#Post activation function
def act(h, beta, activation):
    """
    Activation functions:

    esp  --  expected signal propagation
    relu  -- zero noise limit of esp
    sigma -- sigmoid/Fermi-Dirac distribution

    """
    if activation == 'esp':
        A = h*sigmoid(beta, h)

    elif activation== 'relu':
        A= np.maximum(0,h)

    elif activation== 'sigmoid':
        A= sigmoid(beta, h)

    return A


#--------Forward propagation----------------------------------------------------------------

def FW_prop(X, parameters, activation):

    """
    Arguments:
    X-- placeholder of the input data.
    parameters-- dictionary of parameters, layer by layer, in the network.
    activations-- list of activation functions to apply to the pre-activation outputs

    Evaluates:
    A_prev --activation of the previous layer, used in the fwd pass
    cache_linear["Z"+str(l)]-- dictionary of pre-activation outputs
    cache_act["A"+str(l)]-- dictionary of post-activation outputs

    Returns:
    caches-- array containing all the post and pre- activation values, layer by layer

    """

    cache_linear={} #dictionary, cache of the linear outputs
    cache_act={} #dictionary, cache of activations
    caches = [] #list of output caches

    L= len(parameters)//3+1 # number of layers (this is counted in a different way than later)

    a_prev= X #
    caches.append((X,0) ) #Add the input as the 0th layer. This is used for backpropagation

    for l in range(1,L):

        cache_linear["h"+str(l)] = np.dot(parameters["w"+str(l)], a_prev)+ parameters["b"+str(l)]
        cache_act["a"+str(l)] = act(cache_linear["h"+str(l)], parameters['beta'+str(l)], activation[l-1])
        a_prev= cache_act["a"+str(l)]

        caches.append((cache_act["a"+str(l)], cache_linear["h"+str(l)]))  #Returns all the caches in the FWD pass

    return  caches


#---------------Objective function-----------------------------------------------------------

def obj(AL, Y, activation):

    """
    Arguments:
    AL -- activation of the output layer
    Y --  ground truth.
    activation -- activation function to apply. The last value discriminates between a classificatin (sigmoid) append
    a regression (esp ot relu) taks.

    Returns:
    cost -- cost function
    """

    L= len(activation) #number of layers
    m = Y.shape[1] #number of training examples

    last = activation[L-1]

    if last == 'sigmoid': #use cross entropy loss function
          cost = np.sum(-(np.multiply(Y,np.log(AL))+np.multiply((1-Y),np.log(1-AL)) ))/m

    elif last == 'esp' or last == 'relu': #use minimum squared error (L2 loss)
          cost = np.sum((AL-Y)**2)/(2*m)

    cost = np.squeeze(cost) #to make sure loss is a scalar.

    return cost

#reorder the parameters 
def layer_par(parameters):

    L= int(len(parameters)/3)+1

    W = [parameters['w'+str(l)] for l in range(1,L)]
    Beta = [parameters['beta'+str(l)] for l in range(1,L)]

    return W, Beta

#-------------------------One step of backpropagation-------------------------
def Back_one_step(da, dt, Wl , Betal, hl, a_prev, actl):

    '''
    Evaluate one step of backpropagation:

    Arguments:
    da -- recursive parameters to evaluate dW and db
    dt -- recursive parameters to evaluate dbeta

    Wl    -- weight of the (l-1)th unit
    Betal -- (inverse) noise value of the lth unit
    hl--  pre-activation value of the lth unit
    a_prev--  post-activation value of the previous unit

    Evaluate:
    g -- derivative of the esp at layer l
    f -- derivative of the sigmoid at layer l
    dc -- recursive parameter to compute dW and db for 'esp'
    ds -- recursive parameter to compute dbeta for esp or everything for 'sigmoid'
    da_prev -- value of dA feeding the next layer
    dt_prev -- value of dt feeding the next layer


    Return:
    grads-- dictionary containing the gradients

    '''

    #Initialise variables
    s = sigmoid(Betal, hl) #Activation value of the lth unit.


    if actl == 'esp':
        #Building Blocks
        g = s+hl*Betal*s*(1-s) #derivative of esp w.r.t. w ot b
        f = (hl**2)*s*(1-s) #derivative of esp w.r.t. beta
        dc = da*g

    elif actl == 'sigmoid':
        #Building Blocks
        g = hl*Betal*s*(1-s) #derivative of sigmoid w.r.t. w ot b
        f = hl*s*(1-s) #derivative of sigmoid w.r.t. beta
        dc = da*s*(1-s)

    # gradients of the parameters
    dWl = np.dot(dc,a_prev.T)
    dbl = np.sum(dc, axis=1, keepdims=True)
    dBetal= np.sum(dt*f, axis=1, keepdims=True)

    #Next
    da_prev = np.dot(Wl.T, dc) #update dWl, dbl for the next BW step
    dt_prev = np.dot(Wl.T, (dt*g)) #update dbeta for the next BW step

    assert (da_prev.shape == a_prev.shape)
    assert (dWl.shape == Wl.shape)
    assert (dBetal.shape == Betal.shape)

    return da_prev, dt_prev, dWl, dbl, dBetal

#Back prop through the all network

def BKW_prop(Y, caches, parameters, activation):

    """
    Implement the backward propagation

    Arguments:
    Y -- ground truth
    caches -- list of caches of pre/post activation values of each unit for every layer [1,L]
    parameters -- list of parameters in the model for every layer

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dt" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
             grads["dbeta" + str(l)] = ...
    """

    grads = {}
    a_prev =[]

    L= len(parameters)//3 # number of layers (integer)

    W, Beta = layer_par(parameters) #obtain W and beta for each layer

    m  = Y.shape[1] # number of training examples

    AL = caches[L][0] #post-activation value of the last unit

    actL= activation[L-1]

    assert(Y.shape == AL.shape) #make sure that the dimenstions of the final layer and labels coincides

    # Initializing the backpropagation
    if actL == 'sigmoid':
       delta = - ( np.divide(Y, AL) - np.divide(1 - Y, 1 - AL) )/m

    elif actL == 'esp':
        delta = (AL - Y)/m

    # Initialise the output layer

    grads["da" + str(L)], grads["dt" + str(L)], grads["dw" + str(L)], grads["db" + str(L)], grads["dbeta" + str(L)] = Back_one_step(delta, delta, W[L-1] , Beta[L-1], caches[L][1], caches[L-1][0],actL )

    for l in reversed(range(L-1)):

        da_tmp, dt_tmp ,dWl_tmp, dbl_tmp, dbetal_tmp = Back_one_step( grads["da" + str(l+2)], grads["dt" + str(l+2)], W[l] , Beta[l], caches[l+1][1], caches[l][0] , activation[l])

        grads["da" + str(l + 1)] = da_tmp
        grads["dt" + str(l + 1)] = dt_tmp
        grads["dw" + str(l + 1)] = dWl_tmp
        grads["db" + str(l + 1)] = dbl_tmp
        grads["dbeta" + str(l + 1)] = dbetal_tmp

    return grads

#=================================
#Update rule for the parameters
#=================================
#This is a simple gradient descent update rule. More advanced update rules can be implemented.


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using pure gradient descent

    Arguments:
    parameters --  dictionary containing W, b, beta for each layer
    grads --  dictionary containing the gradients.

    Returns:
    parameters -- python dictionary containing the updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
                  parameters["beta" + str(l)] = ...
    """

    L = len(parameters)//3 # number of layers (integer)

    # Update paramters
    for l in range(L):
        parameters['w'+str(l+1)]= parameters['w'+str(l+1)] - learning_rate* grads["dw" + str(l+1)]

        parameters['b'+str(l+1)]= parameters['b'+str(l+1)] - learning_rate* grads["db" + str(l+1)]
        parameters['beta'+str(l+1)] = parameters['beta'+str(l+1)] - learning_rate* grads["dbeta" + str(l+1)]

    return parameters


#=======================
#Main Program
#=======================


def Run_DNN(x, y, layers, activation, learning_rate, num_iterations, print_cost=False):

    """
    Run the DNN to find the optimal set of paramters

    Arguments:
    X -- data, iput marix
    Y -- true "label" vector
    layers -- list containing the input size and each layer size
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used in the predictor to find the ouput
    """
    #Initialise parameters
    parameters=0

    costs = []                         # store the value of the objective function for each epoch
    gradients= []                      # store the value of the gradients for each epoch

    #Take the transpose of the inputs to abide to our convention
    X = x.T
    Y = y.T

    f , m = X.shape # f: number of features, m: number of training examples

    #Initialise network
    network = np.append(f, layers)
    L= len(network)

    # Parameters initialization.
    parameters= initialize_parameters(network)

    # Loop (gradient stochastic gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation
        caches = FW_prop(X, parameters, activation)

        # Compute cost.
        cost = obj(caches[L-1][0] , Y, activation)

        # Backward propagation. It outputs the value of the gradients for the current epoch. This can be used to study the dynamics
        #of the optimization algo
        grads = BKW_prop(Y, caches, parameters, activation)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 1000 training example
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

        #store the value of cost and grads every 1000 epochs (chanche this value)
        if print_cost and i % 1000 == 0:
            costs.append(cost)
            gradients.append(grads)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters, gradients, costs


#======================
#Prediction functions
#======================

#Prediction and metrics for Regression Models

def predict_reg(x, y, parameters, activation):

    """
    Arguments:
    X-- feature matrix from training/test set
    Y-- ground truth, label matrix   from training/test set
    activation-- array contaiing the type of activation function of each layer
    parameters-- dictionary containing the trained parameters

    Return:
    Yp-- predicted label matrix
    accuracy-- prediction accuracy
    """

    #Take transpose of input
    X = x.T
    Y = y.T

    L = len(parameters)//3
    m = X.shape[1]

    (n_y, m) = y.shape #shape of the ground truth matrix

    caches = FW_prop(X, parameters, activation)

    #An= np.transpose(cache_A["A"+str(L-1)]) #last activation
    Yp = caches[L][0] # predicted output

    accuracy= r2_score(y, Yp.T)

    return  Yp.T , accuracy

#Prediction and metrics for Classification Models

def predict_class(x, y, parameters, activation):
    """
    This function is used to predict the results of a  n-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """
    #Take transpose of input
    X = x.T
    Y = y.T

    L = len(parameters)//3
    m = X.shape[1]
    Yp = np.zeros((1,m), dtype = np.int)

    # Forward propagation
    caches = FW_prop(X, parameters, activation)

    p = caches[L][0] # predicted probability of 0/1

    # convert probs to 0/1 predictions. If p>1/2 Yp=1 otherwise 0. Note that a dynamic threshold coulb be implemented
    for i in range(0, m):
        if p[0,i] > 0.5:
            Yp[0,i] = 1
        else:
            Yp[0,i] = 0

    #evaluate accuracy for single label prediction
    accuracy = np.mean((Yp[0,:] == Y[0,:]) )

    return p, Yp, accuracy

#General Prediction function

def predict(x, y, parameters, activation):

    '''
    Prediction function. It uses either predict_reg or predict_class according to wether the ouput activation
    is esp (relu) or sigmoid

    Returns:
    Predicted value and accuracy. Other metrics can be used than the one proposed here.

    '''

    L = len(parameters)//3

    actL = activation[L-1]

    if actL == 'sigmoid':
        p, Yp, accuracy = predict_class(x, y, parameters, activation)
        model = [p, Yp, accuracy]

    elif actL == 'esp':

        Yp, accuracy = predict_reg(x, y, parameters, activation)
        model = [Yp, accuracy]

    print('accuracy=', accuracy)

    return model
