
##
## Arc Eager Transition Parser
##

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
	REDUCE = 4

	ACTIONS = [ARC_LEFT, ARC_RIGHT, SHIFT, REDUCE]


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
	## Create rel. between head of buffer and top of stack, pop the stack
	## Returns the resulting state
	def arc_left(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		if len(self.stack) < 1 or len(self.buffer) < 1:
			raise Exception("Arc left without items")

		w1 = self.stack[0]
		w2 = self.buffer[0]

		new_rel.append((w1.id, w2.id))
		new_stack.pop(0)

		if self.verbose:
			print("{0} : {1} | ({2} <- {3}) - {4}".format([w.word for w in self.stack], [w.word for w in self.buffer], w2.word, w1.word, "ARC_LEFT"))

		return ArcState(new_buffer, new_stack, new_rel, self.verbose)

	## Perform a right arc transition in this state
	## Create rel. between top of stack and head of buffer, shift head of buffer to top stack
	## Returns the resulting state
	def arc_right(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		if len(self.stack) < 1 or len(self.buffer) < 1:
			raise Exception("Arc right without items")

		w1 = self.stack[0]
		w2 = self.buffer[0]

		new_rel.append((w1.id, w2.id))	## Create rel. between top of stack and head of buffer
		new_buffer.pop(0) 				## Shift head of buffer to top stack
		new_stack.insert(0, w2)

		if self.verbose:
			print("{0} : {1} | ({2} -> {3}) - {4}".format([w.word for w in self.stack], [w.word for w in self.buffer], w2.word, w1.word, "ARC_RIGHT"))

		return ArcState(new_buffer, new_stack, new_rel, self.verbose)

	## Perform a shift transition in this state
	## Remove head of buffer, add to top of stack
	## Returns the resulting state
	def shift(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		## Cannot shift if buffer is empty
		if len(self.buffer) == 0:
			raise Exception("Shift with empty buffer")

		w = new_buffer.pop(0) 	## Remove head of buffer
		new_stack.insert(0, w)	## Add to top of stack

		if self.verbose:
			print("{0} : {1} - {2}".format([w.word for w in self.stack], [w.word for w in self.buffer], "SHIFT"))

		return ArcState(new_buffer, new_stack, new_rel, self.verbose)


	## Perform a reduce transition in this state
	## Pops the stack
	## Returns the resulting state
	def reduce(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		## Cannot reduce if stack is empty
		if len(self.stack) == 0:
			raise Exception("Reduce with empty stack")

		new_stack.pop(0)

		if self.verbose:
			print("{0} : {1} - {2}".format([w.word for w in self.stack], [w.word for w in self.buffer], "REDUCE"))

		return ArcState(new_buffer, new_stack, new_rel, self.verbose)


	## Returns True is this is a valid final state, False otherwise
	def done(self):
		if(len(self.buffer) == 0 and len(self.stack) == 1):
			return True
		else:
			return False

	## Get the next correct action for this state to match the given target graph
	def get_next_action(self, graph):
		if len(self.stack) > 0 and len(self.buffer) > 0:
			s1 = self.stack[0]
			s2 = self.buffer[0]

			if(graph.has_edge(s1.id, s2.id)):
				return ArcState.ARC_LEFT
			elif(graph.has_edge(s2.id, s1.id)):
				return ArcState.ARC_RIGHT
			else:
				for k in range(0, s1.id):
					if (graph.has_edge(k, s2.id) or graph.has_edge(s2.id, k)):
						return ArcState.REDUCE

		elif len(self.stack) > 1:
			return ArcState.REDUCE

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
		if action == ArcState.REDUCE:
			return self.reduce()


	## Return True if the given action can be performed in this state, False otherwise
	def valid_action(self, action):
		if action == ArcState.ARC_LEFT:
			if len(self.stack) < 1 or len(self.buffer) < 1:
				return False
		if action == ArcState.ARC_RIGHT:
			if len(self.stack) < 1 or len(self.buffer) < 1:
				return False
		if action == ArcState.SHIFT:
			if len(self.buffer) == 0:
				return False
		if action == ArcState.REDUCE:
			if len(self.stack) == 0:
				return False

		return True


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

if __name__ == "__main__":
	import sys
	import csv
	import depeval

	file_test = sys.argv[1]
	file_out = sys.argv[2]

	print("Sanity Check")
	k = 0
	with open(file_out, 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)

		## Read graphs from the test file
		for graph in iterCoNLL(file_test):
			## initialize the arcState parser
			arcState = ArcState.initialize_from_graph(graph) 
			#arcState.verbose = True

			try:
				## Keep predicting until we reach an end state
				while not arcState.done():
					arcState = arcState.do_action(arcState.get_next_action(graph))

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

	## Evaluate the resulting output file
	depeval.eval(file_test, file_out)
