import numpy as np
import string
from tqdm import tqdm


class MarkovTextGenerator:
  def __init__(self,filename):
    # retrive raw sentences
    self.sentences = self.read_file(filename=filename)
    # instance distribution placeholder
    self.initial_state_distribution = dict()
    self.first_order_transitions = dict()
    self.second_order_transitions = dict()

  def read_file(self,filename):
    """
    Generic function to extract the sentences in a text file.
    """
    with open(filename) as f:
      sentences = f.readlines()
    f.close()
    return sentences

  def add_2_dict(self,dct,key,value):
    """
    The idea is to use dictionaries instead of matrices in order to counter sparcity. The idea is to have
    a mapping of tokens to list of values to which each token transitions to
    Arguments:
      - dct: Parameters
      - key: the key
      - value: the value
    """
    if key not in dct:
      dct[key] = list()
    dct[key].append(value)

  def sentence_preprocessing(self,sentence):
    """
    Receives a sentence and removes the punctuation as preprocessing
    """
    return sentence.translate(str.maketrans('','',string.punctuation))


  def list_2prob_dict(self,tokens):
    """
    Converts a list of tokens into a dictionary which maps each token to its probability
    Arguments:
      = tokens: List of tokens that model a distribition
    """
    dict_prob = dict()
    N = len(tokens)
    for token in tokens:
      dict_prob[token] = dict_prob.get(token, 0.) + 1
    for token, counts in dict_prob.items():
      dict_prob[token] = counts / N
    return dict_prob

  def normalize_transition_distributions(self,transition_matrix):
    """
      replace list with dictionary of probabilities
    """
    for token,lst in transition_matrix.items():
      transition_matrix[token] = self.list_2prob_dict(lst)
  
  def normalize_initial_distribution(self):
    initial_total = sum(self.initial_state_distribution.values())
    for token,count in self.initial_state_distribution.items():
      self.initial_state_distribution[token] = count / initial_total

  def normalize_distributions(self):
    self.normalize_initial_distribution()

    self.normalize_transition_distributions(
        transition_matrix=self.first_order_transitions
    )

    self.normalize_transition_distributions(
        transition_matrix=self.second_order_transitions
    )


  def populate_distributions(self):
    """
    Populates the enviornment dynamics according to the provided text corpus. 
    """
    for sentence in tqdm(self.sentences): # iterate through all the sentences in the dataset
      parsed_sentence = sentence.rstrip().lower() # normalize the sentence
      if parsed_sentence: # filter out empty sentences
        tokens = self.sentence_preprocessing(parsed_sentence).strip().split() # preprocessing
        T = len(tokens) # get the length of the sentence
        for index,token in enumerate(tokens):
          if index == 0: # we are at the first token of the sentence
            self.initial_state_distribution[token] = self.initial_state_distribution.get(token,0) + 1
          else: # processing for timesteps different than the initial token
            token_1 = tokens[index-1] # retrieve the token at T-1 relative to the current timestep
            if index == T-1: # measure the probability of ending the line
              self.add_2_dict(self.second_order_transitions,(token_1,token),'END')

            if index == 1: # measure the distribution for the first token (fill out the first degree transicion matrix given only the first token)
              self.add_2_dict(self.first_order_transitions,token_1,token)
            else: # measure the standard second order state transition dynamics
              token_2 = tokens[index-2]
              self.add_2_dict(self.second_order_transitions,(token_2,token_1),token)

  def fit(self):
    """
    Applies the training and normalization phases
    """  
    self.populate_distributions()
    self.normalize_distributions()






  def sample_token(self,transition):
    """
    Samples a token from the transition matrix. 
    """
    p0 = np.random.random()
    cumulative = 0
    for token, prob in transition.items():
      cumulative += prob
      if p0 < cumulative:
        return token
    assert(False) # should never get here


  def generate(self,N = 4):
    """
    Takes the distributions and then generates N lines of words.
    """

    for i in range(N):
      sentence = list()
      # sample the initial token
      w_0 = self.sample_token(self.initial_state_distribution)
      sentence.append(w_0)

      # sample second workd 
      w_1 = self.sample_token(self.first_order_transitions[w_0])
      sentence.append(w_1)

      # second order transition generation until we reach the end token
      while True:
        w_2 = self.sample_token(self.second_order_transitions[(w_0,w_1)])
        if w_2 == 'END':
          break
        sentence.append(w_2)
        w_0 = w_1
        w_1 = w_2
      print("[~]" + " " + " ".join(sentence))