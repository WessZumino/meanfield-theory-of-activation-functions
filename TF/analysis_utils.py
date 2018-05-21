'''
Various Utilities
================================================
Author: Mirco Milletari <mirco@bambu.life> (2018)

Collection of functions used for the empirical analysis performed in

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

'''

import numpy as np
import h5py

#Visualizaion
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

#================================
#Hessian Analysis
#================================

#Non uniform epoch sampling

def epoch_sampling(epochs, uniform_step):
    '''
    Defines a non uniform series used in the optimization loop to record the values of (trainin parameters, gradients, hessian)
    during training. This is mostly used to lower the computational needs to evaluate the Hessian matrix. At the beginning of
    training (we take 1/4 of the total epochs), values are recorded more frequently and non-uniformly. The remaining 3/4 of
    training epochs are sampled uniformaly with a costant step.

    Input:
    epochs -- total number of training epochs.
    uniform_step -- uniform step size

    Returns:
    epoch_vec -- a list containing the values of the training epoch at which the system is sampled.

    '''

    #part = int(epochs/5) #defines the size of the non-uniform epoch-sample
    part =200

    vec1=[]
    v=1

    while v< part: #v doubles at each iteration
        vec1.append(v)
        #v = int(2*v)
        v+= 10

    vec2 = list(np.arange(v+10, epochs, step= uniform_step, dtype = 'int')) #uniform part of sampling.

    epoch_vec = vec1+vec2

    return epoch_vec


#save data

def save_all(dict, act, layers):

    costs = dict[0]
    accuracy = dict[1]
    grads_and_vars = dict[2]
    hessians = dict[3]
    residual = dict[4]
    #gs = dict[5]
    index = dict[5]
    eigen = dict[6]
    zeros = dict[7]

    np.save('cost_'+act+str(layers), costs)
    np.save('accuracy_'+act+str(layers), accuracy)
    np.save('grads_and_vars_'+act+str(layers), grads_and_vars)
    np.save('hessians_'+act+str(layers), hessians)
    np.save('residual_'+act+str(layers), residual)
    #np.save('gs_'+act+str(layers), gs)
    np.save('index_'+act+str(layers), index)
    np.save('eigen_'+act+str(layers), eigen)
    np.save('zeros_'+act+str(layers), zeros)



#-------------------------------Index of critical points----------------------------------------------------------

def index(hessian_epoch):

    '''
    Evaluates the normalized index of crtical points as a function of training epochs.

    Inputs:
    hessian_epoch-- a list containing the Hessian matrix evaluated at crtical points, for a given training epoch.

    Returns:
    eigenvalues-- eigenvalues of the Hessian matrix for each training epoch.
    index-- normalized index of critical points, i.e. the average number of negative eigenvalues per training epoch

    zeros-- average number of zero eigenvalues (if any)
    '''

    epochs = len(hessian_epoch) #numbers of entries in the list, corresponding to the numebr of training epochs

    eigenvalues = [np.linalg.eigvals(hessian_epoch[i]) for i in range(epochs)] #list: eigenvalues

    index = [np.mean( 1*(eigenvalues[i] < 0) ) for i in range(epochs)] #list: average number of negative eigenvalues

    zeros = [np.mean( 1*(eigenvalues[i] == 0) ) for i in range(epochs)] # list

    return index, eigenvalues, zeros


#========================
# Visualizaions
#========================

#Plot binary classification datasets
def plot_data(X_train, X_test, y_train, y_test, save):

    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=40, cmap=plt.cm.Spectral)
    plt.scatter(X_test[:, 0],  X_test[:, 1], c=y_test, s=40, cmap=plt.cm.Spectral)
    plt.tick_params(labelsize= 22)

    if save == True: plt.savefig('data.pdf')

    plt.show()


#Plot the indices of critical points
def plot_index(cost, index, zeros, save, layer, act):

    plt.figure(figsize=(8, 6))
    plt.plot(cost, index,linestyle='none' ,marker='o', label= '$\\alpha$')
    plt.plot(cost, zeros,linestyle='none',  marker= '.',label= '$\\gamma$' )
    plt.ylabel('index', size= 'x-large')
    plt.xlabel('$\epsilon$', size = 'x-large')
    plt.legend(ncol=1)
    #plt.title('Index of critical points vs energy:'+str(layer)+','+act)
    if save == True: plt.savefig('index_'+act+'_'+layer+'.pdf')
    plt.show()

def alpha(cost_relu, alpha_relu,cost_esp , alpha_esp,cost_sig, alpha_sig, save):

    plt.figure(figsize=(8, 6))
    plt.plot(cost_relu, alpha_relu,linestyle='none' ,marker='s', label= 'ReLu')
    plt.plot(cost_esp, alpha_esp,linestyle='none',  marker= 'o',label= 'ESP' )
    plt.plot(cost_sig, alpha_sig,linestyle='none',  marker= 'D',label= 'sigmoid' )

    plt.tick_params(labelsize= 22)
    plt.ylabel('$\\alpha$', size= 22)
    plt.xlabel('$\epsilon$', size = 22)
    plt.legend(ncol=1,fontsize= 22, loc= 0)
    plt.ylim((-0.05,0.82))
    plt.xlim((-0.05,0.83))

    if save == True: plt.savefig('alpha.pdf')

    plt.show()

def gamma(cost_relu, gamma_relu, cost_esp, gamma_esp,cost_sig, gamma_sig, save):

    plt.figure(figsize=(8, 6))
    plt.plot(cost_relu, gamma_relu,linestyle='none' ,marker='s', label= 'ReLu')
    plt.plot(cost_esp, gamma_esp,linestyle='none',  marker= 'o',label= 'ESP' )
    plt.plot(cost_sig,  gamma_sig,linestyle='none',  marker= 'D',label= 'sigmoid' )

    plt.tick_params(labelsize= 18)
    plt.ylabel('$\\gamma$', size= 18)
    plt.xlabel('$\epsilon$', size = 18)
    plt.legend(ncol=1, fontsize= 18)
    if save == True: plt.savefig('gamma.pdf')
    plt.show()

#Compare Cost functions
def compare_scalar(esp, relu, sig,epochs, layer, type,save):

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, esp, label='ESP', linestyle='-',linewidth=3)
    plt.plot(epochs, relu, label= 'ReLu', linestyle='--',linewidth=3 )
    plt.plot(epochs, sig, label= 'sigmoid', linestyle='-.',linewidth=3)

    if type ==0:
        plt.ylabel('$\epsilon$', size= 22)
        obj = 'costs'
    else:
        plt.ylabel('$I(\hat{y}- y)$', size= 22)
        obj = 'score'

    #xaxis.set_tick_params(width=5)
    plt.tick_params(labelsize= 22)
    plt.legend(ncol=1,fontsize= 22, loc= 0)
    plt.xlabel('epochs', size= 22)
    plt.ylim((-0.05,0.82))

    if save== True: plt.savefig(obj+'_'+layer+'.pdf')

    plt.show()

#Residues
def compare_res(esp, relu, sig,epochs, layer,save):

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, esp, label='ESP', linestyle='-',linewidth=3)
    plt.plot(epochs, relu, label= 'ReLu', linestyle='--',linewidth=3)
    if sig !=0: plt.plot(epochs, sig, label= 'sigmoid', linestyle='-.',linewidth=3)

    plt.tick_params(labelsize= 22)
    plt.xlabel('epochs', size= 22)
    plt.ylabel('$I(e/m = 0)$', size= 22)
    plt.legend(loc=0, ncol=1, fontsize = 22)
    plt.ylim((-0.05,0.8))
    #plt.xlim((-0.05,0.83))

    if save== True: plt.savefig('residuals_'+layer+'.pdf')

    plt.show()


#Accuracy vs energy
def plot_accuracy(costs, accuracy, save, layer, act):

    plt.figure(figsize=(8, 6))
    plt.plot( costs, accuracy[0], marker='o', label= 'train')
    plt.plot( costs, accuracy[1], marker= '.', label = 'test')
    plt.ylabel('accuracy',size= 'x-large')
    plt.xlabel('$\epsilon$', size= 'x-large')
    plt.legend(ncol=1)
    #plt.title('Accuracy')
    if save == True: plt.savefig('accuracy_'+act+'_'+str(layer)+'.pdf')

    plt.show()


#Plot the distribution of eigenvalues for different training epochs
def distr_epoch_eigenv(epoch, eigen, epoch_sample, layer, save, act):

    ep = epoch_sample[epoch]

    plt.figure(figsize=(8, 6))
    plt.yscale('log', nonposy='clip')
    sns.distplot(a= np.real(eigen[epoch]), hist= True, kde= False ,rug=False, label= 'epoch'+str(ep) )
    plt.legend(ncol=1)
    plt.xlabel('$\lambda$', size= 'x-large')
    plt.ylabel('$\log \, \\rho(\lambda)$', size= 18)
    #plt.title('Hessian eigenvalues distributio'+str(layer)+','+act)

    if save == True: plt.savefig('eigen_'+act+'_'+str(ep)+'_'+str(layer)+'.pdf')
    plt.show()


def distr_epoch_eigenv(epoch, eigen, layer, save, act):

    plt.figure(figsize=(8, 6))
    plt.yscale('log', nonposy='clip')
    sns.distplot(a= np.real(eigen[epoch]), hist= True, kde= False ,rug=False, label= 'epoch'+str(epoch) )
    plt.legend(ncol=1)
    plt.xlabel('$\lambda$', size= 18)
    plt.ylabel('$\log \, \\rho(\lambda)$', size= 18)
    #plt.title('Hessian eigenvalues distributio'+str(layer)+','+act)

    if save == True: plt.savefig('eigen_'+act+'_'+str(layer)+'.pdf')

    plt.show()


#Compare eigenvaue distribution for different activations
def compare_epoch_eigen(epoch, eigen_esp, eigen_relu, eigen_sig, sample, layer, save):

    ep= str(sample[epoch])

    plt.figure(figsize=(8, 6))
    plt.yscale('log', nonposy='clip')
    sns.distplot(a= np.real(eigen_esp[epoch]), hist= True, kde= False ,rug=False, label= 'esp' )
    sns.distplot(a= np.real(eigen_relu[epoch]), hist= True, kde= False ,rug=False, label= 'ReLu' )
    sns.distplot(a= np.real(eigen_sig[epoch]), hist= True, kde= False ,rug=False, label= 'sigmoid' )

    plt.legend(ncol=1, title='epoch='+ep )
    plt.xlabel('$\lambda$', size= 18)
    plt.ylabel('$\log \, \\rho(\lambda)$', size= 18)
    #plt.title('Hessian eigenvalues distributio'+str(layer)+','+act)
    if save == True: plt.savefig('comp_eigen_'+ep+'_'+layer+'.pdf')

    plt.show()



    #f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
    #sns.distplot( eigen[0] , color="skyblue", ax=axes[0, 0])
    #sns.distplot( eigen[1] , color="olive", ax=axes[0, 1])
    #sns.distplot( eigen[2] , color="gold", ax=axes[1, 0])
    #sns.distplot( eigen[3] , color="teal", ax=axes[1, 1])
    #plt.show()

def weight_Relu_esp(i, j, bin, save):

    if j==1:
        k= 0
        m= 0
    elif j==2:
        k=2
        m=3

    d1, d2 = grads_and_vars_relu[0][k][1].shape

    w_relu = np.reshape(grads_and_vars_relu[i][k][1],[-1, d1*d2])
    w_stmp = np.reshape(grads_and_vars_esp[i][m][1],[-1, d1*d2])

    plt.figure(figsize=(8,6))

    sns.distplot( a=w_stmp, hist=True, kde=True, rug=False, bins=bin,color='black', label = 'esp' )
    sns.distplot( a=w_relu, hist=True, kde=True, rug=False, bins=bin, color = 'olive', label= 'ReLu' )

    plt.xlabel('W'+str(j)+' '+ 'values')
    plt.ylabel('Histogram')
    plt.legend(ncol=1)
    plt.title('Distribution of weights 2-50-1: epoch='+str(epoch_vec[i]))
    if save== True: plt.savefig('W'+str(j)+'_'+'str(i)'+'.pdf')

    plt.show()


def parameters_distribution(grads_and_vars, epoch, comp, bin, save):

    d1, d2 = grads_and_vars[epoch][comp][1].shape
    par = np.reshape(grads_and_vars[epoch][comp][1],[-1, d1*d2])

    plt.figure(figsize=(8,6))

    sns.distplot( a=par, hist=True, kde=False,bins = bin, rug=False )

    plt.xlabel('W', size= 'x-large')
    plt.ylabel('Histogram', size= 'x-large')
    plt.legend(ncol=1)
    #if save== True: plt.savefig('W'.pdf')

    plt.show()
