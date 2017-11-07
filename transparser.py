import sys
import csv
import depeval
from collections import defaultdict
from random import shuffle
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

    stack_detailed = [self.graph.node[i.id] for i in self.stack]
    buffer_detailed = [self.graph.node[i.id] for i in self.buffer]
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

    stack_detailed = [self.graph.node[i.id] for i in self.stack]
    buffer_detailed = [self.graph.node[i.id] for i in self.buffer]
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

      stack_detailed = [self.graph.node[i.id] for i in self.stack]
      buffer_detailed = [self.graph.node[i.id] for i in self.buffer]
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
        G.add_node(0, {'word': '*root*', 'lemma': '*root*', 'cpos': '*root*', 'pos': '*root*', 'feats': '*root*'})
        newGraph = False
      [id, word, lemma, cpos, pos, feats, head, drel, phead, pdrel] = l.split('\t')
      G.add_node(int(id), {'word' : word, 'lemma': lemma, 'cpos' : cpos, 'pos'  : pos, 'feats': feats})
      if (head.isdigit()):
        G.add_edge(int(id), int(head))
      b.append(ArcNode(int(id), word))

  if G != None:
    yield {'graph': G, 'buffer': b}
  h.close()


def read_sentences(filename):
  h = open(filename, 'r')
  sentences = []
  current = []
  for line in h:
    line = line.strip()
    if line == "":
      if current:
        sentences.append(current)
        current = []
    else:
      current.append(line.split('\t'))
  return sentences


# Eval function from project 1
import sklearn.metrics
def eval(gold, predicted):
   return (sklearn.metrics.f1_score(gold, predicted, average='micro'), sklearn.metrics.f1_score(gold, predicted, average='macro'))


# For labeled data, read in from CoNLL file and obtain tuples with configurations
def process_labeled_set(filename):
  gen = iterCoNLL(filename)
  configs = []
  for s in gen:
    parser = ArcState(s['buffer'], [ArcNode(0, "*ROOT*")], [], s['graph'], [])
    while not parser.done():
      parser = parser.do_action(parser.get_next_action())
    for config in parser.configs:
      configs.append(config)
  #print "Read " + str(len(configs)) + " configs for " + filename
  return configs


# Obtains list of feature vectors made of config (stack, buffer, transition) tuples
# Implements "basic features" of second part of project
def make_feature_vec(configs):
  factor = 1.
  vec = []
  for config in configs:
    next_vec = defaultdict(float)
    if len(config[0]) > 0:
      attr = config[0][0]
      # Identity of word at top of stack
      next_vec['S_TOP_' + attr['word']] = factor
      # Coarse POS for word at top of stack
      next_vec['S_TOP_POS_' + attr['cpos']] = factor
    if len(config[0]) > 1:
      attr = config[0][1]
      # Indentity of word second in stack
      next_vec['S_SECOND_' + attr['word']] = factor
      # Coarse POS for word second in stack
      next_vec['S_SECOND_POS_' + attr['cpos']] = factor
      attr0 = config[0][0]
      # Pair of words at top of stack (not symmetric)
      next_vec['PAIR_WORDS_' + attr['word'] + '_' + attr0['word']] = factor
      # Pair of POS at top of stack (not symmetric)
      next_vec['PAIR_POS_' + attr['cpos'] + '_' + attr0['cpos']] = factor
    vec.append(next_vec)
  return vec

# Given starting ArcState, uses theta as oracle to run arc-standard transitions
# Returns final ArcState with accumulated transitions from process
def score_with_features(p, theta, transitions):
  while not p.done() and not p.failed:
    scores = defaultdict(float)
    for t in transitions:
      if (len(p.stack) > 0):
        scores[t] += theta[t]['S_TOP_' + p.stack[0].word]
        scores[t] += theta[t]['S_TOP_POS_' + p.graph.node[p.stack[0].id]['cpos']]
      if (len(p.stack) > 1):
        scores[t] += theta[t]['S_SECOND_' + p.stack[1].word]
        scores[t] += theta[t]['S_SECOND_POS_' + p.graph.node[p.stack[1].id]['cpos']]
        scores[t] += theta[t]['PAIR_WORDS_' + p.stack[1].word + '_' + p.stack[0].word]
        scores[t] += theta[t]['PAIR_POS_' + p.graph.node[p.stack[1].id]['cpos'] + '_' + p.graph.node[p.stack[0].id]['cpos']]
    v = list(scores.values())
    k = list(scores.keys())
    yhat = k[v.index(max(v))]
    p = p.do_action(yhat)
  return p


# Given theta weights and feature vector list, predicts transition for each entry
def predict_labels(transitions, theta, feature_vecs):
  labels = []
  for i in range(0, len(feature_vecs)):
    scores = defaultdict(float)
    for transition in transitions:
      for feature in feature_vecs[i]:
        scores[transition] += feature_vecs[i][feature] * theta[transition][feature]
    v = list(scores.values())
    k = list(scores.keys())
    label = k[v.index(max(v))]
    labels.append(label)
  return labels


def perceptron(training_configs, dev_configs, testfile, testout, eval_dev):

  transitions = [ArcState.ARC_LEFT, ArcState.ARC_RIGHT, ArcState.SHIFT]

  # Create beginning structure for weights
  theta = dict()
  m = dict()
  m_last_updated = dict()
  for t in transitions:
    theta[t] = defaultdict(float)
    m[t] = defaultdict(float)
    m_last_updated[t] = defaultdict(float)

  training_vec = make_feature_vec(training_configs)
  dev_vec = make_feature_vec(dev_configs)
  training_labels = []
  for config in training_configs:
    training_labels.append(config[2])
  dev_labels = []
  for config in dev_configs:
    dev_labels.append(config[2])

  # Initialize list of indices used for randomizing training instance order
  indices = []
  for i in range(0, len(training_configs)):
    indices.append(i)

  gen = iterCoNLL("en.dev")
  # Holds tuples of {initial config, gold relation set} for each dev sentence
  dev_golds = []
  for s in gen:
    state = ArcState(s['buffer'], [ArcNode(0, "*ROOT*")], [], s['graph'], [])
    p = state
    while not p.done():
      p = p.do_action(p.get_next_action())
    dev_golds.append({'config': state, 'relations': p.relations})

  # Initialize variables for holding best-performing dev relations
  best_dev_results = 0.
  best_dev_iteration = 0
  best_dev_relations = []
  best_theta = dict()

  # Main perceptron loop
  counter = 0
  iteration = 0
  while (True):

    # Evaluates accuracy on training set after each pass through whole training set
    if (counter % len(training_configs) == 0 and counter > 0):
      m_temp = dict()
      theta_temp = dict()
      for t in transitions:
        m_temp[t] = defaultdict(float)
        theta_temp[t] = defaultdict(float)
        # Obtain weights from running average before evaluating
        for feature in m[t]:
          m_temp[t][feature] = m[t][feature] + theta[t][feature] * (counter - m_last_updated[t][feature])
          theta_temp[t][feature] = m_temp[t][feature] / counter
      #print "Training set config accuracy: " + str(eval(training_labels, predict_labels(transitions, theta_temp, training_vec)))
      #print "Dev set config accuracy: " + str(eval(dev_labels, predict_labels(transitions, theta_temp, dev_vec)))

      correct = 0
      total = 0
      current_dev_relations = []
      for gold in dev_golds:
        p = score_with_features(gold['config'], theta_temp, transitions)
        for relation in gold['relations']:
          total += 1
          if relation in p.relations:
            correct += 1
        current_dev_relations.append(p.relations)
      current_dev_results = float(correct)/total*100
      if (current_dev_results > best_dev_results):
        best_dev_results = current_dev_results
        best_dev_relations = current_dev_relations
        best_dev_iteration = iteration
        best_theta = theta_temp
      elif (iteration - best_dev_iteration >= 5):
        print "Stopping; will use best dev result of " + str(best_dev_results) + " from iteration " + str(best_dev_iteration)
        # Test: write to file dev results
        if eval_dev:
          sentences = read_sentences("en.dev")
          for i in range(0, len(sentences)):
            for j in range(0, len(sentences[i])):
              sentences[i][j][6] = ""
            for relation in current_dev_relations[i]:
              # Writes the head number for the words that have relations
              # Ugly indexing though...
              sentences[i][relation[1]-1][6] = str(relation[0])
          with open("en.dev.out", 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for sentence in sentences:
              for word in sentence:
                writer.writerow(word)
              writer.writerow([])
          depeval.eval("en.dev", "en.dev.out")
        # Code to write to testout results on test sentences
        else:
          gen = iterCoNLL(testfile)
          test_relations = []
          for s in gen:
            state = ArcState(s['buffer'], [ArcNode(0, "*ROOT*")], [], s['graph'], [])
            p = score_with_features(state, best_theta, transitions)
            test_relations.append(p.relations)
          sentences = read_sentences(testfile)
          for i in range(0, len(sentences)):
            for relation in test_relations[i]:
              sentences[i][relation[1]-1][6] = str(relation[0])
          with open(testout, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for sentence in sentences:
              for word in sentence:
                writer.writerow(word)
              writer.writerow([])
        return
      print "Relation attachment score on dev set: " + str(current_dev_results)

      shuffle(indices)
      iteration += 1

    index = indices[counter % len(indices)]

    # Obtain predicted based on argmax of score for each class
    scores = defaultdict(float)
    for t in transitions:
      for feature in training_vec[index]:
        scores[t] += training_vec[index][feature] * theta[t][feature]
    v = list(scores.values())
    k = list(scores.keys())
    yhat = k[v.index(max(v))]

    # If prediction is wrong, update weight vector
    correct_label = training_configs[index][2]
    if (yhat != correct_label):
      for feature in training_vec[index]:
        # Updates for scores of predicted class
        m[yhat][feature] += theta[yhat][feature] * (counter - m_last_updated[yhat][feature])
        m_last_updated[yhat][feature] = counter
        theta[yhat][feature] -= training_vec[index][feature]
        m[yhat][feature] -= training_vec[index][feature]

        # Updates for scores of actual class
        m[correct_label][feature] += theta[correct_label][feature] * (counter - m_last_updated[correct_label][feature])
        m_last_updated[correct_label][feature] = counter
        theta[correct_label][feature] += training_vec[index][feature]
        m[correct_label][feature] += training_vec[index][feature]

    counter += 1

if __name__ == "__main__":

  # Read command line args
  file_train = sys.argv[1]
  file_test = sys.argv[2]
  file_out = sys.argv[3]

  training = process_labeled_set(file_train)
  dev = process_labeled_set("en.dev")

  eval_dev = True
  perceptron(training, dev, file_test, file_out, eval_dev)
