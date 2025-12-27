from builtins import range
from builtins import object
import numpy as np

from cs6353.layers import *
from cs6353.layer_utils import *


class TwoLayerNet(object):
    '''
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    '''

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        '''
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        '''
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        '''
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        '''
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # Forward pass: affine - relu - affine
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        # First layer: affine + ReLU
        out1, cache1 = affine_relu_forward(X, W1, b1)
        
        # Second layer: affine
        scores, cache2 = affine_forward(out1, W2, b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        
        # Backward pass
        grads = {}
        dout1, grads['W2'], grads['b2'] = affine_backward(dscores, cache2)
        dx, grads['W1'], grads['b1'] = affine_relu_backward(dout1, cache1)
        
        # Add L2 regularization gradient
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 normalization=None, reg=0.0, weight_scale=1e-2, dtype=np.float32, seed=None):


        '''
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.

        '''
        self.normalization = normalization
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        

        # Initialize the parameters of the network
        layers_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W' + str(i + 1)] = weight_scale * np.random.randn(layers_dims[i], layers_dims[i + 1]) # set weights here
            self.params['b' + str(i + 1)] = np.zeros(layers_dims[i + 1])

            '''if i == 0:
                # First layer: input_dim (3072) x num_neurons
                self.params['W' + str(i + 1)] = weight_scale * np.random.randn(3072, hidden_dims[i])
                self.params['b' + str(i + 1)] = np.zeros(hidden_dims[i])
            elif i == self.num_layers - 1:
                # Last layer: num_neurons x num_classes (10)
                self.params['W' + str(i + 1)] = weight_scale * np.random.randn(hidden_dims[i - 1], 10)
                self.params['b' + str(i + 1)] = np.zeros(10)
            else:
                # Hidden layers: num_neurons x num_neurons
                self.params['W' + str(i + 1)] = weight_scale * np.random.randn(hidden_dims[i], hidden_dims[i])
                self.params['b' + str(i + 1)] = np.zeros(hidden_dims[i])'''
            if self.normalization and i < self.num_layers - 1:
                self.params['gamma' + str(i + 1)] = np.ones(layers_dims[i + 1])
                self.params['beta' + str(i + 1)] = np.zeros(layers_dims[i + 1])

        # Batch normalization parameters
        self.bn_params = []
        self.ln_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'} for _ in range(self.num_layers)] # -1 
        if self.normalization == 'layernorm':
            self.ln_params = [{} for _ in range(self.num_layers)] # -1

        # Cast parameters to correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    
    def loss(self, X, y=None):
        '''
        #Compute loss and gradient for the fully-connected net.

        #Input / output: Same as TwoLayerNet above.
        '''
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params param since it
        # behaves differently during training and testing.
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        out = X
        caches = []

        for i in range(self.num_layers - 1):
            # Affine forward
            w, b = self.params[f'W{i+1}'], self.params[f'b{i+1}']
            out, fc_cache = affine_forward(out, w, b)

            # Apply batch or layer normalization if needed
            if self.normalization == 'batchnorm':
                gamma, beta = self.params[f'gamma{i+1}'], self.params[f'beta{i+1}']
                out, bn_cache = batchnorm_forward(out, gamma, beta, self.bn_params[i])
            elif self.normalization == 'layernorm':
                gamma, beta = self.params[f'gamma{i+1}'], self.params[f'beta{i+1}']
                out, bn_cache = layernorm_forward(out, gamma, beta, self.ln_params[i])

            # ReLU activation
            out, relu_cache = relu_forward(out)

            # Store caches
            caches.append((fc_cache, bn_cache if self.normalization else None, relu_cache))

        # Last layer: affine without activation
        w, b = self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}']
        scores, fc_cache = affine_forward(out, w, b)
        caches.append(fc_cache)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # Compute loss
        loss, dscores = softmax_loss(scores, y)

        # Add L2 regularization to the loss
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params[f'W{i}'] ** 2)

        # Backward pass
        grads = {}
        dout = dscores

        # Backprop through the last affine layer
        fc_cache = caches.pop()
        dout, dw, db = affine_backward(dout, fc_cache)
        grads[f'W{self.num_layers}'] = dw + self.reg * self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = db

        # Backprop through the remaining layers
        for i in reversed(range(self.num_layers - 1)):
            fc_cache, norm_cache, relu_cache = caches.pop()

            # ReLU backward
            dout = relu_backward(dout, relu_cache)

            # Apply batch or layer normalization backward if needed
            if self.normalization == 'batchnorm':
                dout, dgamma, dbeta = batchnorm_backward(dout, norm_cache)
                grads[f'gamma{i+1}'] = dgamma
                grads[f'beta{i+1}'] = dbeta
            elif self.normalization == 'layernorm':
                dout, dgamma, dbeta = layernorm_backward(dout, norm_cache)
                grads[f'gamma{i+1}'] = dgamma
                grads[f'beta{i+1}'] = dbeta

            # Affine backward
            dout, dw, db = affine_backward(dout, fc_cache)
            grads[f'W{i+1}'] = dw + self.reg * self.params[f'W{i+1}']
            grads[f'b{i+1}'] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
