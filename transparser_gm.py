import networkx as nx
import random 

#from arcStandard import *
from arcEager import *

class ArcStateFeatures():

	def __init__(self, state):
		self.dict = {}

		if (len(state.stack) > 0):
			s1 = state.stack[0]
			#identity of word at top of the stack
			self.dict["s1="+s1.word] = 1.0
			#coarse POS (field 4) of word at top of the stack
			self.dict["s1_pos="+s1.pos] = 1.0

		if (len(state.stack) > 1):
			s2 = state.stack[1]
			#identity of second word in stack
			self.dict["s2="+s2.word] = 1.0
			#coarse POS (field 4) of second word in stack
			self.dict["s2_pos="+s2.pos] = 1.0

		if (len(state.stack) > 1):
			s2 = state.stack[1]
			#pair of words at top of stack
			self.dict["s1_s2="+s1.word+"_"+s2.word] = 1.0
			#pair of coarse POS (field 4) at top of stack
			self.dict["s1_s2_pos="+s1.pos+"_"+s2.pos] = 1.0

		if (len(state.buffer) > 0):
			b1 = state.buffer[0]
			#identity of word at head of buffer
			self.dict["b1="+b1.word] = 1.0
			#coarse POS (field 4) of word at head of buffer
			self.dict["b1_pos="+b1.pos] = 1.0

		if (len(state.buffer) > 1):
			b2 = state.buffer[1]
			#pair of words at head of buffer
			self.dict["b1_b2="+b1.word+"_"+b2.word] = 1.0
			#pair of coarse POS (field 4) at head of buffer
			self.dict["b1_b2_pos="+b1.pos+"_"+b2.pos] = 1.0

		if (len(state.buffer) > 0 and len(state.stack) > 0):
			self.dict["s1_b1="+s1.word+"_"+b1.word] = 1.0
			self.dict["s1_b1_pos="+s1.pos+"_"+b1.pos] = 1.0


	def __iter__(self):
		return self.dict.iteritems()


class Weights(dict):
	## default all unknown feature values to zero
	def __getitem__(self, idx):
		if self.has_key(idx):
			return dict.__getitem__(self, idx)
		else:
			return 0.

	## given a feature vector, compute a dot product
	def dotProduct(self, x):
		dot = 0.
		for feat,val in x:
			dot += val * self[feat]
		return dot

	## Add a weight vector to this weight vector
	def add(self, w):
		for feat,val in w.iteritems():
			self[feat] += val

	## Multiply each entry with a scalar
	def mult(self, s):
		for feat,val in self.iteritems():
			self[feat] = s*val

	# given an example _and_ a true label (y is +1 or -1), update the
	# weights according to the perceptron update rule (we assume
	# you've already checked that the classification is incorrect
	def update(self, x, y):
		for feat,val in x:
			if val != 0.:
				self[feat] += y * val


class Perceptron():
	def __init__(self, classes):
		self.classes = classes

		## Create a weight 'vector' for each class
		self.thetas = []
		for _ in self.classes:
			self.thetas.append(Weights())


	## Train the perceptron with the give input/output pairs
	def train(self, X, Y, maxIter=5, shuffle=False):
		weight_avg = []
		for _ in self.classes:
			weight_avg.append(Weights())

		counter = 1
		ind = range(0, len(X))
		for j in xrange(0, maxIter):
			## Shuffle training examples
			if shuffle:
				random.shuffle(ind)
			correct = 0.
			for i in ind:
				x = X[i]	## A single feature example
				y = Y[i]	## The correct action

				y_ = self._predict(x)[0] 	## The predicted action

				if y_ != y:
					## We made an error and need to update the weights
					index_y_ = self.classes.index(y_)
					index_y = self.classes.index(y)

					self.thetas[index_y_].update(x, -1.0) 	## we predicted this, subtract weight
					self.thetas[index_y].update(x, 1.0) 	## should have been this, add weight

					## Running tally for average weights
					weight_avg[index_y_].update(x, -1.0 * counter)
					weight_avg[index_y].update(x, 1.0 * counter)
				else:
					correct += 1

				counter += 1

			print ("Iteration {0} : {1}".format(j, correct / float(len(X))))
		
		## Compute average weights
		s = 1.0 / counter
		for k in xrange(0, len(weight_avg)):
			weight_avg[k].mult(-s)
			self.thetas[k].add(weight_avg[k])


	## Single perceptron update
	def update(self, arcState, y):
		features = ArcStateFeatures(arcState)
		y_ = self._predict(features)[0]

		if y_ != y:
			## We made an error and need to update the weights
			index_y_ = self.classes.index(y_)
			index_y = self.classes.index(y)

			self.thetas[index_y_].update(features, -1.0) 	## we predicted this, subtract weight
			self.thetas[index_y].update(features, 1.0) 		## should have been this, add weight


	## Return the single best prediction
	def predict(self, arcState):
		features = ArcStateFeatures(arcState)
		actions = self._predict(features)

		return actions[0]

	## Return a list of predictions in descending order of prediction score
	def predict_ordered(self, arcState):
		features = ArcStateFeatures(arcState)
		actions = self._predict(features)

		return actions

	## Return a list of predictions in descending order of prediction score
	def _predict(self, features):
		scores = [0]*len(self.classes)
		for i in xrange(0, len(self.classes)):
			si = self.thetas[i].dotProduct(features)
			scores[i] = si

		## Order the list of actions based on the prediction score
		ordered_pred = [x for _,x in sorted(zip(scores, self.classes), reverse=True)]
		return ordered_pred


def iterCoNLL(filename):
	h = open(filename, 'r')
	G = None
	for l in h:
		l = l.strip()
		if l == "":
			if G != None:
				yield G
			G = None
		else:
			if G == None:
				G = nx.DiGraph()
				G.add_node(0, {'word': '*root*', 'lemma': '*root*', 'cpos': '*root*', 'pos': '*root*', 'feats': '*root*'})
				newGraph = False

			[id, word, lemma, cpos, pos, feats, head, drel, phead, pdrel] = l.split('\t')
			G.add_node(int(id), {'word' : word,
								 'lemma': lemma,
								 'cpos' : cpos,
								 'pos'  : pos,
								 'feats': feats})
			if head != "_":
				G.add_edge(int(id), int(head))

	if G != None:
		yield G
	h.close()


import sys
import csv
import depeval
if __name__ == "__main__":

	## Read arguments from cmd
	file_train = sys.argv[1]
	file_test = sys.argv[2]
	file_out = sys.argv[3]

	## Initialize the perceptron
	perceptron = Perceptron(ArcState.ACTIONS)

	print("Pre-Process")
	## Pre-Process Training Data
	X = []
	Y = []

	## Read graphs from the training file
	for graph in iterCoNLL(file_train):
		## initialize the arcState parser
		arcState = ArcState.initialize_from_graph(graph)

		try:
			while not arcState.done():
				X.append(ArcStateFeatures(arcState))
				action = arcState.get_next_action(graph) 	## Correct action
				Y.append(action)
				arcState = arcState.do_action(action) 		## Update the parser with the correct action

		except Exception as ex:
			pass

	## Train the Perceptron
	print("Start Training")	

	num_iter = 10
	perceptron.train(X, Y, num_iter, shuffle=True)


	## Test
	print("Run Test")
	with open(file_out, 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)

		## Read graphs from the test file
		for graph in iterCoNLL(file_test):

			## initialize the arcState parser
			arcState = ArcState.initialize_from_graph(graph) 

			try:
				## Keep predicting until we reach an end state
				while not arcState.done():
					actions = perceptron.predict_ordered(arcState)
					## Go through the list predicted actions, from best to worst
					## Use the first valid action
					for a in actions:
						if arcState.valid_action(a):
							arcState = arcState.do_action(a)
							break

				## Go though each node in the graph and find the predicted head
				for id in graph.nodes():
					if id == 0:
						continue
					head = 0
					for edge in arcState.relations:
						if edge[1] == id:
							head = edge[0]
							break

					## Write the result to the output file
					word = graph.node[id]['word']
					lemma = graph.node[id]['lemma']
					cpos = graph.node[id]['cpos']
					pos = graph.node[id]['pos']
					feats = graph.node[id]['feats']
					writer.writerow([id, word, lemma, cpos, pos, feats, head, '_', '_', '_'])

			except Exception as ex:
				print(ex)
				pass
			writer.writerow([])
	print("Test Done")

	## Evaluate the resulting output file
	depeval.eval(file_test, file_out)

