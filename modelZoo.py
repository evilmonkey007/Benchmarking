import numpy as np
import theano
import theano.tensor as T
import lasagne

# CONSTANTS 
word2vecDimension = 300

def simpleRecurrentModel(
    word2vec,
    inputVocabSize, 
    batch_size,
    maxGrad,
    hiddenSize,
    ):

    print("Building Model ...")
    
    # Input Layer
    l_in = lasagne.layers.InputLayer((batch_size, None), T.imatrix())
    l_mask = lasagne.layers.InputLayer((batch_size, None), T.bmatrix())

    #Embedding Layer
    l_embedding = lasagne.layers.EmbeddingLayer(incoming=l_in, input_size=inputVocabSize, output_size=word2vecDimension, W=word2vec)

    # Sentence Encoding
    l_encoding = lasagne.layers.GRULayer(
        l_embedding, hiddenSize, mask_input=l_mask, grad_clipping=maxGrad, only_return_final=True)

    # Intermediate Processing Layer
    l_classify = lasagne.layers.DenseLayer(
         l_encoding, num_units=hiddenSize,
         nonlinearity=lasagne.nonlinearities.rectify)

    # Predicting sentiment
    l_out = lasagne.layers.DenseLayer(
        l_classify, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

    return l_in, l_mask, l_out

