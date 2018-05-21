"""
Helper functions for FFN with ESP
=================================================================
Author: Mirco Milletari <mirco@bambu.life> (2018)

Tensorflow implementation of a Feed Forward Deep network with ESP
activation, as defined in

"Expectation propagation: a probabilistic view of Deep Feed Forward Networks"

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

#Tensor Flow
import tensorflow as tf
from tensorflow.python.framework import ops

# ======================================
# Initialize the Computational Graph
# ======================================

#One hot encoding for multiclass classification

def one_hot_econding(vect, N_classes, N_ch):
    """
    One hot encoding:

    For multilcass classification we need to convert the ground truth input vector to a matrix using one hot encoding.

    Labels: Each class appearing in the ground truth vector is encoded in a column vector using: I_i = \Kdelta[i,Y_j] for j in [0, len(Y)],
    where \Kdelta is the kroenecker symbol. As a result, the number of columns in the matrix is equal to N_classes, each column being a binary
    truth tabel: 1 if the text is classified as belonging to book Y_i, 0 if it does not.

    Arguments:
    Y_labels -- ground truth vector
    N_classes -- the number of classes in the ground truth vector
    N_ch -- number of channels, if any  (for the feature vector only)

    Returns:
    one_hot -- one hot matrix encoding
    """

    # Create a tensot flow constant equal to the number of classes
    C = tf.constant(N_classes, name="C")
    one_hot_matrix = tf.one_hot(vect-1, C, axis=0) #axis=0 means it is mapping to column vectors

    if N_ch != 0:
        one_hot_matrix= tf.expand_dims(one_hot_matrix, 1)

    # Create tensodr flow session
    sess = tf.Session()

    vect_hot = sess.run(one_hot_matrix)

    sess.close()

    return vect_hot


#Place Holders for the input/output data

def create_placeholders(Nfeat, Nlab):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    Nfeat -- scalar, size of the feature vector (number of features)
    Nlab  -- scalar, size of the label vector   (number of labels)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    """
    X = tf.placeholder(shape= [Nfeat, None], dtype= "float64" )
    Y = tf.placeholder(shape= [Nlab, None],  dtype= "float64" )

    return X, Y

#parameters initialization

def initialize_parameters(layers, activation, stbeta):

    '''
    Initialise the parameters of the model:

    Arguments:
    layers: Topology of the network. Array contaning number of layers and number of units in each layer.
    activation: list of activation functions, for each layer in the network.

    Evaluate:
    L-- number of layers in the network (excluding the ouput)
    first-- activation of the first layer

    w-- weight matrix, dim: (l, l-1) initialized to a small number drawn from a standard normal distribution
        mean 0 and std 1.
    b-- bias vector, dim: (l,1)
    beta-- inverse "temperature". initialized by sampling from a normal distribution. We Initialise beta small, i.e.
           high temperature. Note that each unit has its own beta as it attains only local equilibrium.
           Another possible initialization of beta is to 1 for each unit.
           Note: If one uses relu as an activation, beta shold be initialized to one and be non trainable.
    initialization:
           Orthogonal weights: tf.initializers.orthogonal()
           Xavier :            tf.contrib.layers.xavier_initializer(seed=1)

    '''

    tf.set_random_seed(1)                   # defines the seed of the random number generator

    parameters={}
    L = len(layers)            # number of layers in the network
    first = activation[0]      #Activation of the first layer


    if first == 'esp':
        train = True
        init = tf.random_normal_initializer(stddev= stbeta)
        #init = tf.ones_initializer()

    else:
         train= False
         init = tf.ones_initializer()


    for l in range(1, L):
        parameters['w' + str(l)] = tf.get_variable('w' + str(l), [layers[l], layers[l-1]],dtype= 'float64' , initializer= tf.contrib.layers.xavier_initializer(seed=1) )
        parameters['b' + str(l)] = tf.get_variable('b' + str(l), [layers[l], 1],dtype= 'float64', initializer = tf.zeros_initializer())
        parameters['beta' + str(l)] = tf.get_variable('beta'+ str(l), [layers[l], 1], dtype= 'float64', initializer = init, trainable= train )


        assert(parameters['w' + str(l)].shape == (layers[l], layers[l-1]))
        assert(parameters['b' + str(l)].shape == (layers[l], 1))
        assert(parameters['beta'+ str(l)].shape == (layers[l], 1))

    return parameters

#Activation functions

def act(h,beta, activation):

    """
    Activation functions:

    esp  -- finite temperature message passing
    relu  -- zero noise limit of esp
    sigma -- Fermi-Dirac distribution

    """

    if activation == "esp" or activation == "softmax":
         A = tf.multiply(h, tf.nn.sigmoid(tf.multiply(beta,h)) )

    elif activation == "sigmoid":
         A = tf.nn.sigmoid(tf.multiply(beta,h))

    elif activation == "relu":
        A = tf.nn.relu(h)

    return A

#--------Forward propagation----------------------------------------------------------------
def FW_prop(X,parameters, activation):

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

    L= len(activation)+1 # number of layers

    a_prev= X

    for l in range(1,L):

        cache_linear["h"+str(l)] = tf.matmul(parameters["w"+str(l)], a_prev)+ parameters["b"+str(l)]
        cache_act["a"+str(l)] = act(cache_linear["h"+str(l)], parameters['beta'+str(l)], activation[l-1])
        a_prev= cache_act["a"+str(l)]

    an =  cache_act["a"+str(L-1)]
    hn =  cache_linear['h'+str(L-1)]

    return an, hn, cache_linear, cache_act

#---------------cost function-----------------------------------------------------------

def obj(zn, betan, Y, activation):

    """
    Arguments:
    zn -- value of the output layer. This can either be equal to the last post activation value for esp and relu
          or the last pre-activation output for sigmoid. This is so because TF autmotically includes the sigmoid
          function in the definition of the cross entropy.

    Y --  ground truth. This needs to be transposed

    Returns:
    cost -- cost function

    """

    L= len(activation) #number of layers

    m = Y.shape[1] #number of training examples

    last = activation[L-1]
    labels= tf.transpose(Y)

    if last == 'sigmoid' or last == 'softmax': #use cross entropy loss function
          logits= tf.transpose(betan*zn[1])
          cost  = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits = logits, multi_class_labels=labels))

    elif last == 'esp' or last == 'relu': #use minimum squared error (L2 loss)
          out = tf.transpose(zn[0])
          cost = tf.reduce_mean(tf.squared_difference(out, labels))/2

    return cost

#------------Hessian-------------------

def flatten(tensor):

    '''
    Flattening function:

    input: a tensor list
    returns: a rank one tensor
    '''

    s= len(tensor) #number of tensors in the list

    for i in range(s):

        dl = tensor[i] #take one element of the gradient list (hence the zero)
        d1, d2 = dl.get_shape() #Obtain tensor dimensions

        fl = tf.reshape(dl,[-1, d1*d2]) #reshape the tensor to a (1, d1*d2) tensor

        #concatenate over all the elemets in the list
        if i==0: flattened = fl # the first time
        else: flattened = tf.concat([flattened, fl], axis=1)

    return flattened

#Hessian
def hessian(grads, par):

    '''
    Evaluates the exact Hessian matrix.
    This function uses the same convention of the Autograd package.

    Inputs:
    grads --- the evaluated gradeints of the cost function

    Returns:
    hessian matrix: a (dim,dim) matrix of second derivatives, where 'dim' is the dimension of
    the flattened gradient tensor.
    '''

    flat_grads = flatten(grads)[0] #flat gradients

    dim = flat_grads.get_shape()[0] #get the dimensions of the flattened tensor

    hess = [] #list

    for i in range (dim):

        dg_i = tf.gradients(flat_grads[i], par) #for each element of grads evaluate the gradients
        dg_i_flat = flatten(dg_i) #flatten the resulting hessian onto a 1 d array
        hess.append(dg_i_flat) #store row by row

    return tf.reshape(hess,[dim, dim]) #returns the reshaped matrix


#=======================
#         Main
#=======================

def Run_DNN(X_train, Y_train, X_test, Y_test, layers, activation, epoch_sample, stdbeta, starter_learning, num_iterations, with_hessian, save_model, Plot):

    """
    Run the DNN to find the optimal set of paramters

    Arguments:
    X -- data, iput marix
    Y -- true "label" vector
    layers -- list containing the input size and each layer size
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    with_hessian -- if true evaluates the exact Hessian matrix at predefinite training intervals
    stdbeta -- standard deviation of the noise paramters for initialization

    Returns:
    costs -- list contaning the value of the cost funciton (energy) at predefinite training intervals

    Training metrics:
        acc_train -- list containing the value of the task specific, training set accuracy at predefinite training intervals
        acc_test -- list containing the value of the task specific, test set accuracy at predefinite training intervals
            task and metrics:
            1) Regression: Returns the R2 score
            2) Binary Classification: Accuracy score
            3) Multiclass Classification: Accuracy score

            Other metrics can be easily implemented, but this is not important for this work.

    gradients_and_par -- list containing the value of the gradients and the training parameters at predefinite training intervals

                         1) The format is: gradients_and_par[a][b][c]; [a] runs over the epochs, [c] in (0,1) selects the
                         gradienst and the parameters respectevely. e.g.  gradients_and_par[5][2][0] returns the value of the gradient
                         of b1 at the 5th entry epoch. The epoch value is predetermined, e.g. one may want to store the results every
                         100 epochs, then [5] -- > 500 epochs.

                         2) [b] runs over the training parameters for each layer. e.g. for a 2 layer network with esp:
                            [0] --> w1, [1] --> b1, [2] --> beta1
                            [3] --> w2, [4] --> b2, [5] --> beta2

                            for Relu, there is no trainable beta, and the indexing [b] is adjusted accordingly.

    Npar -- Total number of trainable unit-paramters in the network. This is printed out during training.

    hessians -- list containing the value of the hessian matrix at predefinite training intervals. The format is
                hessians[a][b][c], where [a] runs over the epoch. For fixed [a], hessians stores the value of the hessian matrix
                evaluated at the critical points; this is a nxn matrix indexed by [b][c]. The size of the matrix is predetermined
                by the number of parameters in the network.

    residuals -- list containing the value of the residuals at predefinite training intervals. As we are only interested in the
                 sign of the residuals, we define it as the difference between the predicted output \hat{y} (an in the code)
                 and the training labels y (Y in the code).
    """

    ops.reset_default_graph()                      # reset the computational graph
    tf.set_random_seed(1)                          # to keep consistent results

    #----------training/test set features-------------------------

    X_tr = np.transpose(X_train)                    # the transpose is taken to adapt to TF convenntion. This is also
    f , m = X_tr.shape                              # f: number of features, m: number of training examples

    X_tst = np.transpose(X_test)                    # the transpose is taken to adapt to TF convenntion. This is also
    _ , mt = X_tst.shape

    #------------Initialise network-------------------------------

    network = np.append(f, layers)                  # add the input layer to the list
    L= len(activation)

    actL = activation[L-1]                          # activation of the last layer. It determines the task

    #-----------training/test set labels-------------------------------

    if actL == 'softmax':
        l= len(np.unique(Y_train))
        Y_tr  = one_hot_econding(Y_train, l,0 )
        Y_tst = one_hot_econding(Y_test,  l,0 )

    else:
        Y_tr = np.transpose(Y_train)                    # how we defined the placeholders.
        Y_tst = np.transpose(Y_test)
        l = Y_tr.shape[0]

    #-----------------initialize parameters of the model--------------------------------------------------------

    X, Y= create_placeholders(f, l)                 # Create Placeholders

    parameters = initialize_parameters(network, activation, stdbeta)
    betan = tf.identity(parameters['beta'+str(L)], name="betan") #add the output noise to the graph for later retrieval

    an, hn, _ , _ = FW_prop(X, parameters, activation) #post and pre-activation output of the last layer

    an = tf.identity(an, name= "an")               #add the output post-activation value to the graph for later retrieval
    hn = tf.identity(hn, name='hn')                #add the output pre-activation value to the graph for later retrieval

    #Create a saver for the Model
    if save_model == True:
        saver = tf.train.Saver()

    #-----------------Initialize the cost and gradients---------------------------------------------------------

    costs = [] #store the cost for different opochs
    cost = obj([an,hn], betan, Y, activation)

    #-----------------Initialize the optimizer-----------------------------------------------------------------
    # Implement an exponential learning rate decay every 1000 epochs

    #Implement a dynamical learning rate
    global_step = tf.Variable(0., trainable=False)
    rate = tf.train.exponential_decay(starter_learning, global_step, 500, 0.9) #exponential learning rate decay
    #rate = starter_learning

    tvars = tf.trainable_variables() #list of trainable variables
    Npar= flatten(tvars).get_shape()[1] #total number of paramters in the network

    print('there are:', Npar,'parameters in the network')

    optimizer  = tf.train.AdamOptimizer(learning_rate = rate)   #Initialize Adam optimizer

    grads_var = optimizer.compute_gradients(cost, tvars ) #Get gradients layer by layer. Note that this function returns the pair (grads, var)
    grads = [grads_var[i][0] for i in range(len(grads_var))] #extract the gradients

    min   = optimizer.apply_gradients(grads_and_vars= grads_var, global_step= global_step)    #Apply the gradients to look for critical points

    gradients_and_par = [] #store gradients and training paramters for different epochs
    hessians = [] #store the hessian for different epochs
    residuals= [] #store the value of the residuals for different epochs
    #gs = []       #store the value of the phase space factor for different epochs

    if with_hessian == True: #if true, it evaluates
        hess =  hessian(grads, tvars) #Hessian matrix
        res  =  tf.subtract(an, Y)    #residual error

    #---------------------------Initialize evaluation metrics----------------------------------------------------
    e_len = len(epoch_sample)

    acc_train = [] #store train accuracy for each epoch
    acc_test  = [] #store test accuracy for each epoch

    if actL == 'sigmoid': #accuracy score for binary class classification

        Yp = tf.greater(an , 0.5)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(Yp, tf.equal(Y,1.0)), "float"))

    elif actL == 'esp' or actL == 'relu': #r2 score

         norm= tf.reduce_mean( tf.squared_difference(Y,tf.reduce_mean(Y)) )
         accuracy = 1 - tf.divide( tf.reduce_mean(tf.squared_difference(an, Y)), norm)

    elif actL == 'softmax': #accuracy score for multiclass classification

         Yp = tf.sigmoid(betan*hn)
         correct = tf.equal(tf.argmax(Yp), tf.argmax(Y))
         accuracy= tf.reduce_mean(tf.cast(correct, "float"))

    #-----------------Initialize the graph and start the session-------------------------------------------------

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)
        jj=0

        for epoch in range(num_iterations):

            _ , epoch_cost, epoch_grad, epoch_acc_train = sess.run([min, cost, grads_var, accuracy], feed_dict={X: X_tr, Y: Y_tr})

            # Print the cost every interval epoch (here uses the inhomogenous interval but you can change it)
            if  jj< e_len and epoch % epoch_sample[jj] == 0:
            #if  epoch % 50 == 0:

                print("Epoch %i, Cost: %f, Train accuracy: %f" % (epoch, epoch_cost,epoch_acc_train))

                costs.append(epoch_cost) #store the costs
                gradients_and_par.append(epoch_grad) #store grads and trainable parameters

                #--------------Store the evaluation metrics------------------------------------
                epoch_acc_test = sess.run(accuracy, feed_dict={X: X_tst, Y: Y_tst})

                acc_test.append(epoch_acc_test)
                acc_train.append(epoch_acc_train)
                #------------------------------------------------------------------------------

                jj+=1 #increase counter

                #---------------------Evaluate and store the Hessian---------------------------
                if with_hessian == True:

                    epoch_hess, epoch_res = sess.run([hess,res], feed_dict={X: X_tr, Y: Y_tr})
                    assert(epoch_hess.shape[1] == Npar) #check the dimensions of the hessian matrix

                    hessians.append(epoch_hess) #store the hessian
                    residuals.append(epoch_res) #store the residuals
                    #gs.append(epoch_g)          #store the gs

                else:
                    hessians.append(1) #returns just ones
                    residuals.append(1)
                    #gs.append(1)

        # plot the cost at the end of training
        if Plot== True:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.title("Learning rate =" + str(starter_learning))
            plt.show()


        print('Train accuracy', acc_train[jj-1])
        print('Test accuracy', acc_test[jj-1])

        accuracy = (acc_train, acc_test)

        if save_model == True:
            saver.save(sess, "saver/esp_model.ckpt")

        sess.close()

        return costs, accuracy, gradients_and_par, hessians, residuals
