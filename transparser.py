import networkx as nx

class ArcNode():
	def __init__(self, id, word, pos):
		self.id = id
		self.word = word
		self.pos = pos

class ArcState():

	## Available Actions
	ARC_LEFT = 1
	ARC_RIGHT = 2
	SHIFT = 3

	def __init__(self, buffer, stack, relations, verbose=False):
		self.buffer = buffer
		self.stack = stack
		self.relations = relations
		self.verbose = verbose

	## Creates the initial state for the parser based on a given graph
	@staticmethod
	def initialize_from_graph(graph):
		b = []
		for id in graph.nodes():
			b.append(ArcNode(id, graph.node[id]['word'], graph.node[id]['cpos']))

		state = ArcState(b, [], [])
		state = state.shift()
		return state


	## Perform a left arc transition in this state
	## Returns the resulting state
	def arc_left(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		if len(self.stack) < 2:
			raise Exception("Arc left without items in stack")

		w1 = self.stack[0]
		w2 = self.stack[1]

		new_rel.append((w1.id, w2.id))
		new_stack.pop(1)

		if self.verbose:
			print("{0} : {1} | ({2} <- {3}) - {4}".format([w.word for w in self.stack], [w.word for w in self.buffer], w2.word, w1.word, "ARC_LEFT"))

		return ArcState(new_buffer, new_stack, new_rel, self.verbose)

	## Perform a right arc transition in this state
	## Returns the resulting state
	def arc_right(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		if len(self.stack) < 2:
			raise Exception("Arc right without items in stack")

		w1 = self.stack[0]
		w2 = self.stack[1]

		new_rel.append((w2.id, w1.id))
		new_stack.pop(0)

		if self.verbose:
			print("{0} : {1} | ({2} -> {3}) - {4}".format([w.word for w in self.stack], [w.word for w in self.buffer], w2.word, w1.word, "ARC_RIGHT"))

		return ArcState(new_buffer, new_stack, new_rel, self.verbose)

	## Perform a shift transition in this state
	## Returns the resulting state
	def shift(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		# If buffer is empty and we need to shift, must be non-projective; return state that is done
		if len(self.buffer) == 0:
			raise Exception("Shift with empty buffer")

		w = self.buffer[0]
		new_buffer.pop(0)
		new_stack.insert(0, w)

		if self.verbose:
			print("{0} : {1} - {2}".format([w.word for w in self.stack], [w.word for w in self.buffer], "SHIFT"))

		return ArcState(new_buffer, new_stack, new_rel, self.verbose)
			

	## Returns True is this is a valid final state, False otherwise
	def done(self):
		if(len(self.buffer) == 0 and len(self.stack) == 1):
			return True
		else:
			return False

	## Get the next correct action for this state to match the given target graph
	def get_next_action(self, graph):
		if len(self.stack) >= 2:
			s1 = self.stack[0]
			s2 = self.stack[1]

			if(graph.has_edge(s2.id, s1.id)):
				return ArcState.ARC_LEFT
			elif(graph.has_edge(s1.id, s2.id)):
				connected = True
				for i,j in graph.edges():
					if(j == s1.id and not ((j,i) in self.relations)):
						connected = False

				if connected:
					return ArcState.ARC_RIGHT

		return ArcState.SHIFT

	## Perform the given action
	## Return the resulting state
	def do_action(self, action):
		if action == ArcState.ARC_LEFT:
			return self.arc_left()
		if action == ArcState.ARC_RIGHT:
			return self.arc_right()
		if action == ArcState.SHIFT:
			return self.shift()


	## Return True if the given action can be performed in this state, False otherwise
	def valid_action(self, action):
		if action == ArcState.ARC_LEFT:
			if len(self.stack) < 2:
				return False
		if action == ArcState.ARC_RIGHT:
			if len(self.stack) < 2:
				return False
		if action == ArcState.SHIFT:
			if len(self.buffer) < 1:
				return False

		return True


class ArcStateFeatures():

	def __init__(self, state):
		self.dict = {}

		if (len(state.stack) > 0):
			s1 = state.stack[0]
			#identity of word at top of the stack
			self.dict["s1="+s1.word] = 1.0
			#coarse POS (field 4) of word at top of the stack
			self.dict["s1_pos="+s1.pos] = 1.0

		if (len(state.buffer) > 0):
			b1 = state.buffer[0]
			#identity of word at head of buffer
			self.dict["b1="+b1.word] = 1.0
			#coarse POS (field 4) of word at head of buffer
			self.dict["b1_pos="+b1.pos] = 1.0

		if (len(state.stack) > 1):
			s2 = state.stack[1]
			#pair of words at top of stack
			self.dict["s1_s2="+s1.word+"_"+s2.word] = 1.0
			#pair of coarse POS (field 4) at top of stack
			self.dict["s1_s2_pos="+s1.pos+"_"+s2.pos] = 1.0

		#if (len(state.buffer) > 1):
		#	b2 = state.buffer[1]
		#	#pair of words at head of buffer
		#	self.dict["b1_b2="+b1.word+"_"+b2.word] = 1.0
		#	#pair of coarse POS (field 4) at head of buffer
		#	self.dict["b1_b2_pos="+b1.pos+"_"+b2.pos] = 1.0


	def __iter__(self):
		return self.dict.iteritems()


class Weights(dict):
	# default all unknown feature values to zero
	def __getitem__(self, idx):
		if self.has_key(idx):
			return dict.__getitem__(self, idx)
		else:
			return 0.

	# given a feature vector, compute a dot product
	def dotProduct(self, x):
		dot = 0.
		for feat,val in x:
			dot += val * self[feat]
		return dot

	# given an example _and_ a true label (y is +1 or -1), update the
	# weights according to the perceptron update rule (we assume
	# you've already checked that the classification is incorrect
	def update(self, x, y):
		for feat,val in x:
			if val != 0.:
				#self[feat] += y * val
				self[feat] += y


class Perceptron():
	def __init__(self, classes):
		self.classes = classes
		self.thetas = []
		for _ in self.classes:
			self.thetas.append(Weights())


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
	nn = 0
	for l in h:
		l = l.strip()
		if l == "":
			if G != None:
				yield G
			G = None
		else:
			if G == None:
				nn = nn + 1
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
	actions = [ArcState.ARC_LEFT, ArcState.ARC_RIGHT, ArcState.SHIFT]
	perceptron = Perceptron(actions)
	num_iter = 10

	## Train
	print("Start Training")	
	for i in xrange(0, num_iter):
		correct = 0.
		total = 0.

		## Read graphs from the training file
		for graph in iterCoNLL(file_train):
			## initialize the arcState parser
			arcState = ArcState.initialize_from_graph(graph)

			try:
				while not arcState.done():
					p = perceptron.predict(arcState)			## Current prediction
					action = arcState.get_next_action(graph) 	## Correct action
					
					## Count correct predictions to estimate progress
					if action == p:
						correct += 1
					total += 1

					## Perceptron update step
					perceptron.update(arcState, action)
					arcState = arcState.do_action(action) 		## Update the parser with the correct action

			except Exception as ex:
				pass

		print ("Iteration {0} : {1}".format(i, correct / total))
	print("Training Done")

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
					writer.writerow([id, graph.node[id]['word']] + ['_']*4 + [head] + ['_']*3 )

			except Exception as ex:
				print(ex)
				pass
			writer.writerow([])
	print("Test Done")

	## Evaluate the resulting output file
	depeval.eval(file_test, file_out)

