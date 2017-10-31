from collections import defaultdict
from random import shuffle
from transparser import *

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
      #print "Stack: " + str(config[0])
      #print "Buffer: " + str(config[1])
      #print "Transition: " + str(config[2])
      configs.append(config)

  print "Appeared to succeed with all configs for " + filename
  print len(configs)

  return configs


# Obtains list of feature vectors based on lists of config tuples
# Note: config tuples are (stack, buffer, transition)
# Currently implements "basic features" of second part of project
# TODO: Add extra features for later in project
# TODO: See if performance changes with symmetric pairs
def make_feature_vec(configs):
  factor = 1
  vec = []
  for config in configs:
    next_vec = defaultdict(int)
    if len(config[0]) > 0:
      attr = config[0][0]
      # Identity of word at top of stack
      next_vec['S_TOP_' + attr['word']] = factor
      # Coarse POS for word at top of stack
      next_vec['S_TOP_POS_' + attr['cpos']] = factor
    if len(config[1]) > 0:
      attr = config[1][0]
      # Indentity of word at head of buffer
      next_vec['B_HEAD_' + attr['word']] = factor
      # Coarse POS for word at head of buffer
      next_vec['B_HEAD_POS_' + attr['cpos']] = factor
    if len(config[0]) > 0 and len(config[1]) > 0:
      attr0 = config[0][0]
      attr1 = config[1][0]
      # Pair of words at top of stack and head of buffer (currently not symmetric)
      next_vec['PAIR_WORDS_' + attr0['word'] + '_' + attr1['word']] = factor
      # Pair of POS at top of stack and head of buffer (currently not symmetric)
      next_vec['PAIR_POS_' + attr0['cpos'] + '_' + attr1['cpos']] = factor
    vec.append(next_vec)
  return vec


# Given theta weights and feature vector list, predicts transition for each entry
def predict_labels(transitions, theta, feature_vecs):
  labels = []
  for i in range(0, len(feature_vecs)):
    scoring = defaultdict(int)
    for transition in transitions:
      for feature in feature_vecs[i]:
        scoring[transition] += feature_vecs[i][feature] * theta[transition][feature]
    v = list(scoring.values())
    k = list(scoring.keys())
    label = k[v.index(max(v))]
    labels.append(label)
  return labels


# TODO: Currently has various pieces adapted from project 1 that need changing
# TODO: Primary issue: has no stopping condition; should have one based in some way on accuracy of dev set parsing; fix!
# TODO: Should run on test set before quitting when training has completed
def perceptron(training_configs, dev_configs):

  transitions = [ArcState.ARC_LEFT, ArcState.ARC_RIGHT, ArcState.SHIFT]

  # Create beginning structure for weights
  theta = dict()
  m = dict()
  m_last_updated = dict()
  for t in transitions:
    theta[t] = defaultdict(int)
    m[t] = defaultdict(int)
    m_last_updated[t] = defaultdict(int)

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

  # Main perceptron loop
  counter = 0
  while (True):

    # Evaluates accuracy on training set after each pass through whole training set
    if (counter % len(training_configs) == 0 and counter > 0):
      m_temp = dict()
      theta_temp = dict()
      for t in transitions:
        m_temp[t] = defaultdict(int)
        theta_temp[t] = defaultdict(int)
        # Obtain weights from running average before evaluating
        for feature in m[t]:
          m_temp[t][feature] = m[t][feature] + theta[t][feature] * (counter - m_last_updated[t][feature])
          theta_temp[t][feature] = m_temp[t][feature] / counter
      print "Config transition accuracy on training set: " + str(eval(training_labels, predict_labels(transitions, theta_temp, training_vec)))

      predicted_labels = predict_labels(transitions, theta_temp, dev_vec)
      dev_results = eval(dev_labels, predicted_labels)
      print "Config transition accuracy on dev set: " + str(dev_results)
    
      # TODO: Use dev_golds and theta to obtain relation accuracy
      correct = 0
      total = 0
      for gold in dev_golds:
        p = gold['config']
        while not p.done() and not p.failed:
          scoring = defaultdict(int)
          for t in transitions:
            # TODO: As add extra features, add here too
            if (len(p.stack) > 0):
              scoring[t] += theta[t]['S_TOP_' + p.stack[0].word]
              scoring[t] += theta[t]['S_TOP_POS_' + p.graph.node[p.stack[0].id]['attr_dict']['cpos']]
            if (len(p.buffer) > 0):
              scoring[t] += theta[t]['B_HEAD_' + p.buffer[0].word]
              scoring[t] += theta[t]['B_HEAD_POS_' + p.graph.node[p.buffer[0].id]['attr_dict']['cpos']]
            if (len(p.stack) > 0 and len(p.buffer) > 0):
              scoring[t] += theta[t]['PAIR_WORDS_' + p.stack[0].word + '_' + p.buffer[0].word]
              scoring[t] += theta[t]['PAIR_POS_' + p.graph.node[p.stack[0].id]['attr_dict']['cpos'] + '_' + p.graph.node[p.buffer[0].id]['attr_dict']['cpos']]
          v = list(scoring.values())
          k = list(scoring.keys())
          # TODO: In case of ties, which transition should be default?
          # This choice seems important...
          yhat = k[v.index(max(v))]
          p = p.do_action(yhat)
        for relation in gold['relations']:
          total += 1
          if relation in p.relations:
            correct += 1
      print "Relation attachment score on dev set: " + str(float(correct)/total*100)

      shuffle(indices)

    index = indices[counter % len(indices)]

    # Obtain predicted based on argmax of score for each class
    scoring = defaultdict(int)
    for t in transitions:
      for feature in training_vec[index]:
        scoring[t] += training_vec[index][feature] * theta[t][feature]
    v = list(scoring.values())
    k = list(scoring.keys())
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

  training = process_labeled_set("en.tr100")
  dev = process_labeled_set("en.dev")

  perceptron(training, dev)
