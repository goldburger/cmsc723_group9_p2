from transparser import *

def perceptron():
  return

if __name__ == "__main__":

  gen = iterCoNLL("en.tr100")

  # For now, just work with first one from generator
  first = next(gen)

  parser = ArcState(first['buffer'], [ArcNode(0, "*ROOT*")], [], first['graph'], [], verbose=True)
  while not parser.done():
    parser = parser.do_action(parser.get_next_action())

  for config in parser.configs:
    print "Stack: " + str(config[0])
    print "Buffer: " + str(config[1])
    print "Transition: " + str(config[2])
