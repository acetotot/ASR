import math
import numpy as np
from utils.text_utils import Mapper

def recLabelingProb(t, s, mat, labelingWithBlanks, cache):
	"""recursively compute probability of labeling, save results of sub-problems in cache to avoid recalculating them"""
	blank = 0
	# check index of labeling
	if s < 0:
		return 0.0

	# sub-problem already computed
	if cache[t][s] != None:
		return cache[t][s]

	# initial values
	if t == 0:
		if s == 0:
			res = mat[0, blank]
		elif s == 1:
			res = mat[0, labelingWithBlanks[1]]
		else:
			res = 0.0

		cache[t][s] = res
		return res

	# recursion on s and t
	res = (recLabelingProb(t-1, s, mat, labelingWithBlanks, cache) +
		   recLabelingProb(t-1, s-1, mat, labelingWithBlanks, cache)) \
		  * mat[t, labelingWithBlanks[s]]

	# in case of a blank or a repeated label, we only consider s and s-1 at t-1, so we're done
	if labelingWithBlanks[s] == blank or (s >= 2 and labelingWithBlanks[s-2] == labelingWithBlanks[s]):
		cache[t][s] = res
		return res

	# otherwise, in case of a non-blank and non-repeated label, we additionally add s-2 at t-1
	res += recLabelingProb(t-1, s-2, mat, labelingWithBlanks, cache) * mat[t, labelingWithBlanks[s]]
	cache[t][s] = res
	return res


def emptyCache(maxT, labelingWithBlanks):
	"""create empty cache"""
	return [[None for _ in range(len(labelingWithBlanks))] for _ in range(maxT)]


def ctcLabelingProb(mat, gt):
	"""calculate probability p(gt|mat) of a given labeling gt and a matrix mat according to section 'The CTC Forward-Backward Algorithm' in Graves paper"""
	maxT, _ = mat.shape  # size of input matrix
	blank = 0
	labelingWithBlanks = extendByBlanks(gt, blank)  # ground truth text as label string extended by blanks
	cache = emptyCache(maxT, labelingWithBlanks)  # cache subresults to avoid recalculating subproblems
	r = recLabelingProb(maxT-1, len(labelingWithBlanks)-1, mat, labelingWithBlanks, cache) + \
		recLabelingProb(maxT-1, len(labelingWithBlanks)-2, mat, labelingWithBlanks, cache)
	return r, cache


def ctcLoss(mat, gt):
	"""calculate CTC loss"""
	try:
		P, cache = ctcLabelingProb(mat, gt)
		return -math.log(P), cache
	except Exception as e:
		print(e)
		return float('inf')


def extendByBlanks(seq, b):
	"""extends a label seq. by adding blanks at the beginning, end and in between each label"""
	res = [b]
	for s in seq:
		res.append(s)
		res.append(b)
	return res


if __name__ == "__main__":
	file = ["hamburger.wav", "hoburger.wav", "han_ba_ge.wav", "zz_hamburger.wav", "noise.wav", "null.wav"]
	mapper = Mapper()
	y = mapper.encode_word("hamburger")
	for f in file:
		log_prob = np.load(F"log_prob/{f.replace('wav', 'npy')}").squeeze(1)
		log_prob = np.exp(log_prob)
		loss, cache = ctcLoss(log_prob, y)
		print(loss)
		cache = np.asarray(cache)
		cache[cache == None] = 0
		import pandas as pd
		cache = pd.DataFrame(cache)
		cache.to_excel("cache.xlsx")
		print(cache.shape)


