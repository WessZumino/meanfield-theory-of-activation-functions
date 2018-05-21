"""
Multi-class classification problem for the MNIST with ESP activation (tensorflow implementation)
=====================================================================================================
Author: Mirco Milletari <mirco@bambu.life> (2018)

Multi-class classification problem for the MNIST dataset

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
import matplotlib.pyplot as plt

from esp_tf_utils import* #Helper functions used in Run_DNN()
from analysis_utils import* #Contains functions for plotting and analysis

#Data science libraries
from sklearn.model_selection import train_test_split
from sklearn import datasets

np.random.seed(1)


#This is used in interactive mode. Comment these two lines when running from shell
%load_ext autoreload
%autoreload 2


#==================================
#Data Loading and processing
#==================================
#Load dataset: 8x8 images of 10 digits
digits = datasets.load_digits()

X = digits.images #images
Y= digits.target #labels

images_and_labels = list(zip(X, Y))

for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(1,4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
    plt.savefig('mnist.pdf')

X_flat = X.reshape(X.shape[0], -1)/255 #Flatten and normalize the image
n_labels = len(np.unique(Y)) #number of labels in the dataset

#Split train/test set
X_train, X_test, y_train, y_test = train_test_split(X_flat, Y, test_size=0.2, random_state=0)

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

Various functions for plotting are defined in the analysis_utils file.

'''

layers = [25, n_labels]
epochs = 2000 #number of recurrence steps for gradient descent
epoch_sample= epoch_sampling(epochs, 50) #implement non uniform sampling
l= len(epoch_sample)


#------------------------Relu Network----------------------------------
activation_relu = np.append(['relu']*(len(layers)-1), ['softmax'])

costs_relu, accuracy_relu, grads_and_vars_relu, hessians_relu, residual_relu = Run_DNN(X_train, y_train, X_test, y_test, layers,
        activation_relu, epoch_sample, stdbeta=0.04, starter_learning=0.01, num_iterations= epochs, with_hessian= True, save_model= False, Plot= True)


index_relu, eigen_relu, zeros_relu =  index(hessians_relu)

dict_par = [costs_relu, accuracy_relu, grads_and_vars_relu, hessians_relu, residual_relu, index_relu, eigen_relu, zeros_relu]
save_all(dict_par,'relu', layers)

#----------------------------------------FTMP Network---------------------------------------------------------------------------

activation_ftmp= np.append(['esp']*(len(layers)-1), ['softmax']) # change this unit according to the task.

costs_esp, accuracy_esp, grads_and_vars_esp, hessians_esp, residual_esp = Run_DNN(X_train, y_train, X_test, y_test, layers,
        activation_ftmp, epoch_sample, stdbeta= 0.004, starter_learning=0.01, num_iterations= epochs, with_hessian= True, save_model= False, Plot= True)


index_esp, eigen_esp, zeros_esp =  index(hessians_esp)

dict_par = [costs_esp, accuracy_esp, grads_and_vars_esp, hessians_esp, residual_esp, index_esp, eigen_esp, zeros_esp]
save_all(dict_par,'esp', layers)


#------------------------------------Sigmoid--------------------------------------------------------------------

activation_sig= np.append(['sigmoid']*(len(layers)-1), ['softmax']) # change this unit according to the task.

costs_sig, accuracy_sig, grads_and_vars_sig, hessians_sig, residual_sig = Run_DNN(X_train, y_train, X_test, y_test, layers,
        activation_sig, epoch_sample, stdbeta= 0.04, starter_learning=0.01, num_iterations= epochs, with_hessian= True, save_model= False, Plot= True)

index_sig, eigen_sig, zeros_sig =  index(hessians_sig) #evaluates the normalized index, eigenvalues and cheks for zeros

dict_par = [costs_sig, accuracy_sig, grads_and_vars_sig, hessians_sig, residuals_sig, index_sig, eigen_sig, zeros_sig]
save_all(dict_par,'sigmoid', layers)
