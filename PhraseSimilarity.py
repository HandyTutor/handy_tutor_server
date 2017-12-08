import gensim
from gensim.models import word2vec
pathToBinVectors = './data/file.bin'

print ("Loading the data file... Please wait...")
model1 = gensim.models.KeyedVectors.load_word2vec_format(pathToBinVectors, binary=True)
print ("Successfully loaded 3.6 G bin file!")

import numpy as np
import math
from scipy.spatial import distance

from random import sample
import sys
from nltk.corpus import stopwords


class PhraseVector:
	def __init__(self, phrase):
		self.vector = self.PhraseToVec(phrase)
	def ConvertVectorSetToVecAverageBased(self, vectorSet, ignore = []):
		if len(ignore) == 0: 
			return np.mean(vectorSet, axis = 0)
		else: 
			return np.dot(np.transpose(vectorSet),ignore)/sum(ignore)

	def PhraseToVec(self, phrase):
		cachedStopWords = stopwords.words("english")
		phrase = phrase.lower()
		wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
		vectorSet = []
		for aWord in wordsInPhrase:
			try:
				wordVector=model1[aWord]
				print(wordVector.shape)
				print(wordVector)
				vectorSet.append(wordVector)

			except:
				pass
		return self.ConvertVectorSetToVecAverageBased(vectorSet)

	def CosineSimilarity(self, otherPhraseVec):
		cosine_similarity = np.dot(self.vector, otherPhraseVec) / (np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
		try:
			if math.isnan(cosine_similarity):
				cosine_similarity=0
		except:
			cosine_similarity=0		
		return cosine_similarity

def SimilarityScore(input1, input2):
	phraseVector1 = PhraseVector(input1)
	phraseVector2 = PhraseVector(input2)		
	similarityScore  = phraseVector1.CosineSimilarity(phraseVector2.vector)
	return similarityScore

print(SimilarityScore('boy', 'girl'))
