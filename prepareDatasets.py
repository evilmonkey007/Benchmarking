import HTMLParser
from unidecode import unidecode
import re
import cPickle
import pdb
import os
from nltk.tokenize import word_tokenize
import codecs
import numpy as np
import word_vec_google

# DATASET: Fine: SST1, Coarse: SUBJ, SST2, RT, IMDB
datasetsToLocation = {
	'SUBJ' : 'Datasets/SUBJ.pkl',
	'SST1' : 'Datasets/SST1.pkl',
	'SST2' : 'Datasets/SST2.pkl',
	'RT' : 'Datasets/RT.pkl',
	'IMDB' : 'Datasets/IMDB.pkl',
}

# DATASET FORMAT: 
# (A) Matrix with 2 columns. 
# 		Column 1: Cleaned string
# 		Column 2: Label of string
# (B) Dictionary mapping words in the corpus to a unique ID (May contain <RARE> tag for rare words)
# (C) TODO: Word2Vec matrix associated with dictionary.

# Function to clean string
def processString(string):
	# To take care of:
	# 	's (separate from word), &#** ; (html unescape), /, -, ''

	# Rules (Heuristic) 
	string = re.sub(r'-+', r'-', string) # converting multiple hyphens to single
	string = re.sub(r'([-,/])(\w+)', r' \1 \2', string) # separating hyphenated phrases
	string = re.sub(r'<br /><br />', ' -BR- ', string) # for IMDB

	# HTML unescape (for SUBJ)
	string = re.sub(r'&#(\d+) ;', r'&#\1;', string) 
	string = HTMLParser.HTMLParser().unescape(string)

	# Remove accents (for SUBJ)
	string = unidecode(string)

	# Tokenizing to help with punctutation
	string = " ".join(word_tokenize(string))
	string = re.sub("\'(\w{2,})", "\' \\1", string)

	# Mapping Numbers to a <NUMBER> tag
	string = re.sub(r"\d+", "<NUMBER>", string)

	# To Lower Case
	string = string.lower()
	return string

def getDictionary(wordCount, minFrequency=2):

	# Words with frequency < minFrequency represented by <RARE>
	dictionaryByWordCount = {'<RARE>': 0}

	# Mapping words to a unique ID sorted by their frequency
	for idx, (key, value) in enumerate(sorted(wordCount.items(), key = lambda x: (-x[1], x[0]))):
		if (value < minFrequency):
			break

		dictionaryByWordCount[key] = idx + 1

	return dictionaryByWordCount

def prepareSST():
	# Whole sentences only. Not considering phrases
	# Read sentences from dataset.txt
	sentencesInCorpus = []
	reSentence = re.compile(r'\d+\t(.+)')
	with codecs.open('Datasets/stanfordSentimentTreebank/datasetSentences.txt', 'r', 'utf8') as file:
		file.readline()
		for line in file:
			string = reSentence.match(line).group(1).strip()
			string = re.sub(r'-LRB-', '(', string)
			string = re.sub(r'-RRB-', ')', string)
			sentencesInCorpus.append(string)

	# Read dictionary (mapping from phrase to index)
	sentenceDict = {}
	reMap = re.compile(r'(.+)\|(\d+)')
	with codecs.open('Datasets/stanfordSentimentTreebank/dictionary.txt', 'r', 'utf8') as file:
		for line in file:
			sentenceToIndex = reMap.match(line)
			sentenceDict[sentenceToIndex.group(1).strip()] = int(sentenceToIndex.group(2).strip())

	# Read sentiment (mapping from index to sentiment)
	sentimentDict = {}
	reMap = re.compile(r'(\d+)\|(0(\.\d+)?|1)')
	with codecs.open('Datasets/stanfordSentimentTreebank/sentiment_labels.txt', 'r', 'utf8') as file:
		file.readline()
		for line in file:
			indexToSentiment = reMap.match(line)
			sentimentDict[int(indexToSentiment.group(1).strip())] = float(indexToSentiment.group(2).strip())

	# SST-1 (fine grained)
	corpusSST1 = []
	wordCount = {}
	for sentence in sentencesInCorpus:
		corpusSST1.append((processString(sentence).lower(), sentimentDict[sentenceDict[sentence]]))

		for word in processString(sentence).lower().split():
			wordCount[word] = wordCount.get(word, 0) + 1

	dictionaryByWordCount = getDictionary (wordCount)
	cPickle.dump((corpusSST1, dictionaryByWordCount), open('Datasets/SST1.pkl', 'wb'))

	# SST-2 (positive (1) if >= 0.6 and negative (0) if < 0.4)
	corpusSST2 = []
	for sentence in sentencesInCorpus:
		sentiment = sentimentDict[sentenceDict[sentence]]
		if (sentiment >= 0.6 or sentiment < 0.4):
			corpusSST2.append((processString(sentence).lower(), int(round(sentiment))))

	cPickle.dump((corpusSST2, dictionaryByWordCount), open('Datasets/SST2.pkl', 'wb'))

def prepareIMDB():
	# Label '1': POSITIVE, '0': NEGATIVE
	# Notes: Multiple sentences per example, 

	corpus = []
	wordCount = {}

	def readFiles(directoryName, label):

		for filename in os.listdir(directoryName):
			with codecs.open(os.path.join(directoryName, filename), 'r', encoding='utf8') as file:
				string = processString(file.read().strip().lower())

				corpus.append((string, label))
				for word in string.split():
					wordCount[word] = wordCount.get(word, 0) + 1

	readFiles('Datasets/aclImdb/train/pos', 1)
	readFiles('Datasets/aclImdb/test/pos', 1)
	readFiles('Datasets/aclImdb/train/neg', 0)
	readFiles('Datasets/aclImdb/test/neg', 0)

	dictionaryByWordCount = getDictionary (wordCount)

	cPickle.dump((corpus, dictionaryByWordCount), open('Datasets/IMDB.pkl', 'wb'))

def prepareSUBJorRT(isRT):
	# SUBJ: Label '1': SUBJECTIVE, '0': OBJECTIVE
	# Note: Some reviews in Spanish, unexpected HTML attributes

	# RT: Label '1': POSITIVE, '0': NEGATIVE

	corpus = []
	wordCount = {}

	def readFile(filename, label, encoded):

		with codecs.open(filename, 'r', encoding=encoded) as file:
			for line in file:
				string = processString(line.strip())

				corpus.append((string, label))
				for word in string.split():
					wordCount[word] = wordCount.get(word, 0) + 1

	if (isRT):
		readFile('Datasets/rt-polaritydata/rt-polarity.pos', 1, 'windows-1252')
		readFile('Datasets/rt-polaritydata/rt-polarity.neg', 0, 'windows-1252')
	else:
		readFile('Datasets/rotten_imdb/quote.tok.gt9.5000', 1, 'utf8')
		readFile('Datasets/rotten_imdb/plot.tok.gt9.5000', 0, 'utf8')


	dictionaryByWordCount = getDictionary(wordCount)

	if (isRT):
		cPickle.dump((corpus, dictionaryByWordCount), open(datasetsToLocation['RT'], 'wb'))
	else:
		cPickle.dump((corpus, dictionaryByWordCount), open(datasetsToLocation['SUBJ'], 'wb'))


def browseDataset(dataset):
	corpus, dictionaryByWordCount = cPickle.load(open(datasetsToLocation[dataset], 'rb'))
	pdb.set_trace()

# Create a random set of indices for train, valid and test according to input ratio
def splitCorpus(corpusLength, split = (0.8, 0.1, 0.1)):
	train, valid, _ = ((np.array(split, dtype=float)/sum(split))*corpusLength).astype(int)
	indices = np.random.permutation(range(corpusLength))

	return indices[:train], indices[train:train + valid], indices[train + valid:]

def getWordVecs(vocabToID):
	word2vec, wordsNotInWord2vec = word_vec_google.load_bin_vec("GoogleNews-vectors-negative300.bin", vocabToID)
	word_vec_google.add_unknown_words(word2vec, vocabToID, wordsNotInWord2vec)

	print ("Words in vocab: {}, Words not in word2vec: {}".format(len(vocabToID), len(wordsNotInWord2vec)))
	return word2vec

if __name__ == "__main__":
	browseDataset('RT')
	# string = '\'Hi\' is a weird word.'
	# processString(string)