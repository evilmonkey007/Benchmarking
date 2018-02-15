import modelZoo
import prepareDatasets

import numpy as np
import theano
import theano.tensor as T
import lasagne # for NN functions
import cPickle # for loading and saving
import os # for file directory operations
import time # for timing the training
import math # for NaN

import pdb

# TODO: Standard naming convention (underscores for variable names?)
# TODO: Create cross-validation framework
# TODO: Create framework to save results.

# List of possible optimizers
optimizers = {
	"adagrad": lasagne.updates.adagrad,
	"rmsprop": lasagne.updates.rmsprop,
	"adadelta": lasagne.updates.adadelta,
	"adam": lasagne.updates.adam,
}

# List of hyperparameters to optimise for (dataset, model): hiddenSize, regConstant, Optimizer(?)
hyperparameters = [
	'hiddenSize',
	'regConstant',
	'optimizer',
]

def trainModel(
	dataset,
	model,
	batchSize = 64,
	maxGrad = 100,
	hiddenSize = 200,
	regConstant = 10e-5,
	optimizer = lasagne.updates.adadelta,
	modelPath = None,
	patience = 5,
	):


	def runEpoch(corpus, splitIndices, dictionary, batchSize, NNFunction):

		def createMiniBatch(corpus, dictionary):

			def longestSentenceLength(corpus):
				return max([len(sentence.split()) for sentence, _ in corpus])

			# Initialising Matrices
			corpusMatrix = np.zeros((len(corpus), longestSentenceLength(corpus)), dtype=np.int32)
			maskMatrix = np.zeros_like(corpusMatrix, dtype=np.int8)
			targetVector = np.zeros(len(corpus))

			# Populating matrices
			for sentenceIndex, (sentence, sentiment) in enumerate(corpus):
				for wordIndex, word in enumerate(sentence.split()):
					corpusMatrix[sentenceIndex, wordIndex] = dictionary.get(word, 0) # Assigns word as rare (ID 0) if not in dictionary
					maskMatrix[sentenceIndex, wordIndex] = 1
				targetVector[sentenceIndex] = sentiment

			return corpusMatrix, maskMatrix, targetVector

		np.random.shuffle(splitIndices)
		avgError = 0.0
		avgLoss = 0.0

		for idx in xrange(len(splitIndices)//batchSize):
			batchIndices = splitIndices[ idx*batchSize : min((idx + 1)*batchSize, len(splitIndices))]
			corpusMatrix, maskMatrix, targetVector = createMiniBatch(corpus[batchIndices], dictionary)

			predictedValues, loss = NNFunction(corpusMatrix, targetVector, maskMatrix)

			if(math.isnan(loss)):
				print "Nan found"
				exit()

			# Calculating loss and error
			predictedValues = np.around(predictedValues)
			avgError += np.sum(np.logical_xor(targetVector,np.squeeze(predictedValues)))
			avgLoss += loss


		#normalizing for number of batches
		avgError = avgError/len(splitIndices)
		avgLoss = avgLoss/(len(splitIndices)//batchSize)

		return avgError, avgLoss

	# READ DATA
	corpus, dictionaryByWordCount = cPickle.load(open(prepareDatasets.datasetsToLocation[dataset], 'rb'))
	trainIndices, validIndices, testIndices = prepareDatasets.splitCorpus(len(corpus))
	word2vec = prepareDatasets.getWordVecs(dictionaryByWordCount)
	corpus = np.array(corpus)

	# MODEL CREATION
	# Build Model
	l_in, l_mask, l_out = model(word2vec, len(dictionaryByWordCount), batchSize, maxGrad, hiddenSize)
	
	# Reload weights, if needed
	# TODO: Standardise save file format
	if (modelPath is not None):
		_, _, params = cPickle.load(open(modelPath, 'rb'))
		lasagne.layers.set_all_param_values(l_out, params)

	# Get output Tensor
	network_output = lasagne.layers.get_output(l_out)

	# Define output labels
	target_values = T.vector('target_output')

	# The network output will have shape (batchSize, 1); let's flatten to get a
	# 1-dimensional vector of predicted values
	predicted_values = network_output.flatten()

	# Define loss
	if dataset == 'SST1':
		loss = T.mean(lasagne.objectives.binary_hinge_loss(predicted_values, target_values))
	else:
		loss = T.mean(lasagne.objectives.binary_crossentropy(predicted_values, target_values))

	# regularization
	matrixRegularizaiton = lasagne.regularization.regularize_network_params(l_out, lasagne.regularization.l2)
	
	# final cost
	cost = loss + regConstant*matrixRegularizaiton

	# Retrieve all parameters from the network
	all_params = lasagne.layers.get_all_params(l_out, trainable=True) 

	# Updates
	updates = optimizer(cost, all_params)
	
	# Functions 
	print("Compiling functions ...")
	train = theano.function([l_in.input_var, target_values, l_mask.input_var], [predicted_values, loss], updates=updates)
	predictOutputs = theano.function( [l_in.input_var, target_values, l_mask.input_var],  [predicted_values, loss])

	# TRAINING THE NETWORK
	numEpochs = 0
	badCounter = 0
	trainTime = 0
	saveFileLocation = 'Model Dumps/{}-{}'.format(dataset, model.__name__)
	bestModel = None
	bestExistingModel = None

	# Checking for best existing model for (Model, Dataset) pair
	if (os.path.isfile(saveFileLocation)):
		bestExistingModel = cPickle.load(open(saveFileLocation, 'rb'))

	while (patience > badCounter):
		numEpochs = numEpochs + 1
		badCounter = badCounter + 1

		# Computing train, valid, test errors
		startTime = time.time()
		trainError, trainCost = runEpoch(corpus, trainIndices, dictionaryByWordCount, batchSize, train)
		trainTime += time.time() - startTime

		validError, validCost = runEpoch(corpus, validIndices, dictionaryByWordCount, batchSize, predictOutputs)

		testError, testCost = runEpoch(corpus, testIndices, dictionaryByWordCount, batchSize, predictOutputs)

		# Printing results
		print "******************************************"
		print "Epoch " + str(numEpochs) + " "
		print "******************************************"
		print "Training Total Incorrect and Loss: " + str(trainError) + " " + str(trainCost) 
		print "Validation Total Incorrect and Loss: " + str(validError) + " " + str(validCost) 
		print "Test Total Incorrect and Loss: " + str(testError) + " " + str(testCost) 

		# Checking if best model
		if (bestModel is None or validError < bestModel[0]):
			bestModel = (validError, [locals()[param] for param in hyperparameters], lasagne.layers.get_all_param_values(l_out))
			badCounter = 0

	print "Training Time (Per Epoch): " + str(trainTime/numEpochs)

	# Saving if better than existing model's dump
	if (bestExistingModel is None or bestModel[0] < bestExistingModel[0]):
		cPickle.dump(bestModel, open(saveFileLocation, 'wb'))

if __name__ == '__main__':

	trainModel('RT', modelZoo.simpleRecurrentModel)
