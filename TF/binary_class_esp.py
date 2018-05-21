"""
Binary classification problem with ESP activation (tensorflow implementation)
==============================================================================
Author: Mirco Milletari <mirco@bambu.life> (2018)

Binary classification problem analyzed in

"Expectation propagation: a probabilistic view of Deep Feed Forward Networks"

See 'esp_tf_utils' for definitions of the algorithm and descriptions of the
entries of the main function Run_DNN().

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

import numpy as np
import scipy.io

from esp_tf_utils import* #Helper functions used in Run_DNN()
from analysis_utils import* #Contains functions for plotting and analysis

#Data science libraries
from sklearn.model_selection import train_test_split
np.random.seed(1)

#This is used in interactive mode. Comment these two lines when running from shell
%load_ext autoreload
%autoreload 2

#==================================================
#Data Loading: Choose only one of the two datasets
#==================================================

data = scipy.io.loadmat('NLdata.mat') #Load one of the two provided datasets.

X, Y = (data['X'], data['y'])
m, _ = X.shape
n_labels= Y.shape[1]

#Split train/test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

plot_data(X_train, X_test, y_train, y_test, save= False)


#=====================================
# Implementation
#=====================================

'''
Set up the network topology: the ouput layer is fixed by the number of labels in the ground truth vector Y.

We use a non-uniform sampling when evaluating the Hessian to speed up calculations. At the beginning of
training the Hessian is evaluated for shorter epoch intervals, as this is were most of the changes happen.
For the rest of training, we use homogenous epoch intervals of 50. See the definition of the function in
the analysis_utils file.

For each activation we run the network and evaluate for sampled training epochs:

Cost (loss) function, train/test set accuracy, gradients and learning paramters, Hessian matrix, residuals.

The definition of each of thes above quantities can be found in the esp_tf_utils file.

The index(hessian) function is defined in analysis_utils: it evaluates the eigenvalues of the Hessian and then
the alpha and gamma indices.

Various functions for plotting are also used and defined in the analysis_utils file.


'''


layers = [10, n_labels] #Fix the network topology.
epochs = 2300 #number of recurrence steps for gradient descent.
epoch_sample= epoch_sampling(epochs, 50) #implement non uniform sampling. The second argument is the homogeneous step interval
l= len(epoch_sample)

#--------------------------------------------Relu Network-----------------------------------------------------

activation_relu = np.append(['relu']*(len(layers)-1), ['sigmoid'])

costs_relu, accuracy_relu, grads_and_vars_relu, hessians_relu, residual_relu = Run_DNN(X_train, y_train, X_test, y_test, layers,
        activation_relu, epoch_sample, stdbeta=0.0001, starter_learning=0.01, num_iterations= epochs, with_hessian= False, save_model= False, Plot= True)

index_relu, eigen_relu, zeros_relu =  index(hessians_relu)

dict_par = [costs_relu, accuracy_relu, grads_and_vars_relu, hessians_relu, residual_relu, index_relu, eigen_relu, zeros_relu]
save_all(dict_par,'relu', 8_2)


plot_index(costs_relu,index_relu, zeros_relu, save= True, layer= 8_2, act= 'relu')

plot_accuracy(costs_relu, accuracy_relu, save=True, layer= layers, act= 'relu')

#----------------------------------------ESP Network---------------------------------------------------------------------------

activation_esp= np.append(['esp']*(len(layers)-1), ['sigmoid']) # change this unit according to the task.

costs_esp, accuracy_esp, grads_and_vars_esp, hessians_esp, residual_esp = Run_DNN(X_train, y_train, X_test, y_test, layers,
        activation_esp, epoch_sample, stdbeta= 0.004, starter_learning=0.01, num_iterations= epochs, with_hessian= False, save_model= False, Plot= True)


index_esp, eigen_esp, zeros_esp =  index(hessians_esp)

dict_par = [costs_esp, accuracy_esp, grads_and_vars_esp, hessians_esp, residual_esp, index_esp, eigen_esp, zeros_esp]
save_all(dict_par,'esp', 82)


plot_index(costs_esp,index_esp, zeros_esp, save= True, layer= 8_2, act= 'esp')

plot_accuracy(costs_esp, accuracy_esp, save=True, layer= layers, act= 'esp')

#-------------------------------------Sigmoid------------------------------------------------------------------------

activation_sig= np.append(['sigmoid']*(len(layers)-1), ['sigmoid']) # change this unit according to the task.

costs_sig, accuracy_sig, grads_and_vars_sig, hessians_sig, residual_sig = Run_DNN(X_train, y_train, X_test, y_test, layers,
        activation_sig, epoch_sample, stdbeta= 0.01, starter_learning=0.01, num_iterations= epochs, with_hessian= False, save_model= False, Plot= True)

index_sig, eigen_sig, zeros_sig = index(hessians_sig) #evaluates the normalized index, eigenvalues and cheks for zeros

dict_par = [costs_sig, accuracy_sig, grads_and_vars_sig, hessians_sig, residual_sig, index_sig, eigen_sig, zeros_sig]
save_all(dict_par,'sigmoid', 82)


plot_index(costs_sig,index_sig, zeros_sig, save= True, layer= 82, act= 'sigmoid')

plot_accuracy(costs_sig, accuracy_sig, save=False, layer= layers, act= 'esp')

#======================
# Comparison Analysis
#======================

#Cost function
compare_scalar(costs_esp, costs_relu, costs_sig,epoch_sample, '10', type=0, save=True)

#Train set score
compare_scalar(accuracy_esp[0], accuracy_relu[0], accuracy_sig[0], epoch_sample, layers, type=1, save=True)


#--------------------------Eigenvalue distributions------------------------------------------------

compare_epoch_eigen(10, eigen_esp, eigen_relu, eigen_sig,epoch_sample, layers, save=True)

#--------------------------Residues---------------------

res_esp = [np.mean(residual_esp[i]) for i in range(l)]
res_relu = [np.mean(residual_relu[i]) for i in range(l)]
res_sig = [np.mean(residual_sig[i]) for i in  range(l)]

m1= residual_esp[0].shape[1]

zero_res_esp = [np.sum(1*(np.abs(residual_esp[i]) <= 10**(-5)))/m1 for i in range(l)]
zero_res_relu = [np.sum(1*(np.abs(residual_relu[i]) <= 10**(-5)))/m1 for i in range(l)]
zero_res_sig =  [np.sum(1*(np.abs(residual_sig[i])  <= 10**(-5)))/m1 for i  in range(l)]

compare_res(zero_res_esp, zero_res_relu, zero_res_sig, epoch_sample, '10', save= False)
