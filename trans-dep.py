from transparser import *

def perceptron():
  return

if __name__ == "__main__":

  gen = iterCoNLL("en.tr100")

  training = []
  i = 0
  for s in gen:
    parser = ArcState(s['buffer'], [ArcNode(0, "*ROOT*")], [], s['graph'], [])
    while not parser.done():
      parser = parser.do_action(parser.get_next_action())
    for config in parser.configs:
      #print "Stack: " + str(config[0])
      #print "Buffer: " + str(config[1])
      #print "Transition: " + str(config[2])

      training.append(config)
    print i
    i += 1

  print "Appeared to succeed with all configs for training"
