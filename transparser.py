import networkx as nx

class ArcNode():
	def __init__(self, id, word):
		self.id = id
		self.word = word


class ArcState():

	## Available Actions
	ARC_LEFT = 1
	ARC_RIGHT = 2
	SHIFT = 3

	def __init__(self, buffer, stack, relations, graph, configs, verbose=False):
		self.buffer = buffer
		self.stack = stack
		self.relations = relations
		self.verbose = verbose
		self.graph = graph
		## List for the configuration + action items to be used for training
		self.configs = configs
    ## Value used to check if failed when trying to make a transition
		self.failed = False

	## Perform a left arc transition in this state
	## Returns the resulting state
	def arc_left(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		if len(self.stack) < 2:
			self.failed = True
			return self

		w1 = self.stack[0]
		w2 = self.stack[1]

		new_rel.append((w1.id, w2.id))
		new_stack.pop(1)

		if self.verbose:
			print("{0} : {1} | ({2} <- {3}) - {4}".format([w.word for w in self.stack], [w.word for w in self.buffer], w2.word, w1.word, "ARC_LEFT"))

 		stack_detailed = [self.graph.node[i.id]['attr_dict'] for i in self.stack]
 		buffer_detailed = [self.graph.node[i.id]['attr_dict'] for i in self.buffer]
 		self.configs.append((stack_detailed, buffer_detailed, ArcState.ARC_LEFT))

		return ArcState(new_buffer, new_stack, new_rel, self.graph, self.configs, self.verbose)

	## Perform a right arc transition in this state
	## Returns the resulting state
	def arc_right(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		if len(self.stack) < 2:
			self.failed = True
			return self

		w1 = self.stack[0]
		w2 = self.stack[1]

		new_rel.append((w2.id, w1.id))
		new_stack.pop(0)

		if self.verbose:
			print("{0} : {1} | ({2} -> {3}) - {4}".format([w.word for w in self.stack], [w.word for w in self.buffer], w2.word, w1.word, "ARC_RIGHT"))

 		stack_detailed = [self.graph.node[i.id]['attr_dict'] for i in self.stack]
 		buffer_detailed = [self.graph.node[i.id]['attr_dict'] for i in self.buffer]
 		self.configs.append((stack_detailed, buffer_detailed, ArcState.ARC_RIGHT))

		return ArcState(new_buffer, new_stack, new_rel, self.graph, self.configs, self.verbose)

	## Perform a shift transition in this state
	## Returns the resulting state
	def shift(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		# If buffer is empty and we need to shift, must be non-projective; return state that is done
		if len(self.buffer) > 0:

			w = self.buffer[0]
			new_buffer.pop(0)
			new_stack.insert(0, w)

			if self.verbose:
				print("{0} : {1} - {2}".format([w.word for w in self.stack], [w.word for w in self.buffer], "SHIFT"))

 			stack_detailed = [self.graph.node[i.id]['attr_dict'] for i in self.stack]
 			buffer_detailed = [self.graph.node[i.id]['attr_dict'] for i in self.buffer]
 			self.configs.append((stack_detailed, buffer_detailed, ArcState.SHIFT))
    
			return ArcState(new_buffer, new_stack, new_rel, self.graph, self.configs, self.verbose)

		else:
			self.failed = True
			self.buffer = []
			self.stack = [""]
			return self

	## Returns True is this is a valid final state, False otherwise
	def done(self):
		if(len(self.buffer) == 0 and len(self.stack) == 1):
			return True
		else:
			return False

	## Get the next correct action for this state to match the given target graph
	def get_next_action(self):
		if len(self.stack) >= 2:
			s1 = self.stack[0]
			s2 = self.stack[1]

			if(self.graph.has_edge(s2.id, s1.id)):
				return ArcState.ARC_LEFT
			elif(self.graph.has_edge(s1.id, s2.id)):
				connected = True
				for i,j in self.graph.edges():
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



def iterCoNLL(filename):
    h = open(filename, 'r')
    G = None
    b = []
    nn = 0
    for l in h:
        l = l.strip()
        if l == "":
            if G != None:
                yield {'graph': G, 'buffer': b}
            G = None
            b = []
        else:
            if G == None:
                nn = nn + 1
                G = nx.DiGraph()
                G.add_node(0, attr_dict={'word': '*root*', 'lemma': '*root*', 'cpos': '*root*', 'pos': '*root*', 'feats': '*root*'})
                newGraph = False
            # TODO: Record all parts of split line in graph, to allow accessing for extra features for latter part of project
            [id, word, lemma, cpos, pos, feats, head, drel, phead, pdrel] = l.split('\t')
            G.add_node(int(id), attr_dict={'word' : word,
                                 'lemma': lemma,
                                 'cpos' : cpos,
                                 'pos'  : pos,
                                 'feats': feats})
            G.add_edge(int(id), int(head)) # 'true_rel': drel, 'true_par': int(id)})
            b.append(ArcNode(int(id), word))

    if G != None:
        yield {'graph': G, 'buffer': b}
    h.close()


if __name__ == "__main__":

	b = [ArcNode(1, "book"), \
	ArcNode(2, "the"), \
	ArcNode(3, "flight"), \
	ArcNode(4, "through"), \
	ArcNode(5, "Huston")]

	testGraph = nx.DiGraph()
	testGraph.add_node(0, attr_dict={'word': '*root*',   'pos': '*root*'})
	testGraph.add_node(1, attr_dict={'word': 'book',      'pos': 'DT'})
	testGraph.add_node(2, attr_dict={'word': 'the',    'pos': 'JJ'})
	testGraph.add_node(3, attr_dict={'word': 'flight',  'pos': 'NN'})
	testGraph.add_node(4, attr_dict={'word': 'through',      'pos': 'VB'})
	testGraph.add_node(5, attr_dict={'word': 'Huston',    'pos': 'JJ'})

	testGraph.add_edge(1, 0)   # root -> book
	testGraph.add_edge(3, 1)   # book -> flight
	testGraph.add_edge(2, 3)   # the <- flight
	testGraph.add_edge(5, 3)   # flight -> Huston
	testGraph.add_edge(4, 5)   # through <- Huston

	parser = ArcState(b, [ArcNode(0, "root")], [], testGraph, [], verbose=True)
	while not parser.done():
		parser = parser.do_action(parser.get_next_action())

	print("Done")


	"""
	b = [ArcNode(1, "the"), \
	ArcNode(2, "hairy"), \
	ArcNode(3, "monster"), \
	ArcNode(4, "ate"), \
	ArcNode(5, "tasty"), \
	ArcNode(6, "little"), \
	ArcNode(7, "children")]

	parser = ArcState(b, [ArcNode(0, "root")], [], verbose=True)

	testGraph = nx.DiGraph()
	testGraph.add_node(0, attr_dict={'word': '*root*',   'pos': '*root*'})
	testGraph.add_node(1, attr_dict={'word': 'the',      'pos': 'DT'})
	testGraph.add_node(2, attr_dict={'word': 'hairy',    'pos': 'JJ'})
	testGraph.add_node(3, attr_dict={'word': 'monster',  'pos': 'NN'})
	testGraph.add_node(4, attr_dict={'word': 'ate',      'pos': 'VB'})
	testGraph.add_node(5, attr_dict={'word': 'tasty',    'pos': 'JJ'})
	testGraph.add_node(6, attr_dict={'word': 'little',   'pos': 'JJ'})
	testGraph.add_node(7, attr_dict={'word': 'children', 'pos': 'NN'})
	testGraph.add_edge(1, 3)   # the -> monster
	testGraph.add_edge(2, 3)   # hairy -> monster
	testGraph.add_edge(3, 4)   # monster -> ate
	testGraph.add_edge(4, 0)   # ate -> root
	testGraph.add_edge(5, 7)   # tasty -> children
	testGraph.add_edge(6, 7)   # little -> children
	testGraph.add_edge(7, 4)   # children -> ate
	"""

	"""
	b = [ArcNode(1, "book"), ArcNode(2, "me"), ArcNode(3, "the"), ArcNode(4, "morning"), ArcNode(5, "flight")]
	parser = ArcState(b, [ArcNode(0, "root")], [], verbose=True)

	parser = parser.shift()
	parser = parser.shift()
	parser = parser.arc_right()
	parser = parser.shift()
	parser = parser.shift()
	parser = parser.shift()
	parser = parser.arc_left()
	parser = parser.arc_left()
	parser = parser.arc_right()
	parser = parser.arc_right()
	parser = parser.done()
	"""
