import numpy as np
import pdb


def load_bin_vec(fname, vocabToID): #taken from Madhu
	"""
	Loads 300x1 word vecs from Google (Mikolov) word2vec
	"""

	word_vecs = np.zeros((len(vocabToID), 300))
	wordsNotInWord2vec = set(vocabToID.keys())

	with open(fname, "rb") as f:
		header = f.readline()
		vocab_size, layer1_size = map(int, header.split())
		binary_len = np.dtype('float32').itemsize * layer1_size
		for line in xrange(vocab_size):
			word = []
			while True:
				ch = f.read(1)
				if ch == ' ':
					word = ''.join(word)
					break
				if ch != '\n':
					word.append(ch)   
			if word in vocabToID:
			   word_vecs[vocabToID[word]] = np.fromstring(f.read(binary_len), dtype='float32')
			   wordsNotInWord2vec.remove(word)
			else:
				f.read(binary_len)
	return word_vecs, wordsNotInWord2vec

def add_unknown_words(word_vecs, vocabToID, wordsNotInWord2vec, k=300): #taken from madhu
	"""
	For words that occur in at least min_df documents, create a separate word vector.	
	0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
	"""

	for word in wordsNotInWord2vec:
		word_vecs[vocabToID[word]] = np.random.uniform(-0.25,0.25,k) 


