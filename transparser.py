
class ArcNode():
	def __init__(self, id, word):
		self.id = id
		self.word = word


class ArcBasicState():
	def __init__(self, buffer, stack, relations, verbose=False):
		self.buffer = buffer
		self.stack = stack
		self.relations = relations
		self.verbose = verbose


	def arcLeft(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		w1 = self.stack[0]
		w2 = self.stack[1]

		new_rel.append((w1.id, w2.id))
		new_stack.pop(1)

		if self.verbose:
			print("{0} : {1} | ({2} <- {3})".format([w.word for w in self.stack], [w.word for w in self.buffer], w2.word, w1.word))

		return ArcBasicState(new_buffer, new_stack, new_rel, self.verbose)

	def arcRight(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		w1 = self.stack[0]
		w2 = self.stack[1]

		new_rel.append((w2.id, w1.id))
		new_stack.pop(0)

		if self.verbose:
			print("{0} : {1} | ({2} -> {3})".format([w.word for w in self.stack], [w.word for w in self.buffer], w2.word, w1.word))

		return ArcBasicState(new_buffer, new_stack, new_rel, self.verbose)

	def shift(self):
		new_buffer = list(self.buffer)
		new_stack = list(self.stack)
		new_rel = list(self.relations)

		w = self.buffer[0]
		new_buffer.pop(0)
		new_stack.insert(0, w)

		if self.verbose:
			print("{0} : {1})".format([w.word for w in self.stack], [w.word for w in self.buffer]))

		return ArcBasicState(new_buffer, new_stack, new_rel, self.verbose)

	def done(self):
		if(len(self.buffer) == 0 and len(self.stack) == 1):
			print("Done!")
		else:
			print("No you are not done")


if __name__ == "__main__":
	b = [ArcNode(1, "book"), ArcNode(2, "me"), ArcNode(3, "the"), ArcNode(4, "morning"), ArcNode(5, "flight")]
	parser = ArcBasicState(b, [ArcNode(0, "root")], [], verbose=True)

	parser = parser.shift()
	parser = parser.shift()
	parser = parser.arcRight()
	parser = parser.shift()
	parser = parser.shift()
	parser = parser.shift()
	parser = parser.arcLeft()
	parser = parser.arcLeft()
	parser = parser.arcRight()
	parser = parser.arcRight()
	parser = parser.done()

