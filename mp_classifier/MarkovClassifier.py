import numpy as np


class MarkovTextClassifier:
  def __init__(self,data,vocabulary):
    """
    This models works with the outputs of the PoemDataset Class
    Parameters:
      - data: A dictionary containing list for raw training data, the tokenized version of the training data, and the corresponding labels
      - vocabulary: A dictionary containing the token to index and reverse dictionaries and the labels to class mapping as well.
    """
    self.vocabulary = vocabulary
    self.data = data
    self.v_size = len(vocabulary["vocab"]) # stores the vocabulary size, which defines the dimentions for the state transition matrix and the initial state distribution
    self.k_size = len(vocabulary["label_2_class"]) # number of classes
    self.params = dict() # dictionary containing parameters (state transition matrix and initial state distribution for all classes)
    self.priors = dict() # dictionaly containing the prior probabilities


  def fit(self):
    self.compute_prior_distributions()
    self.cluster_parameters_initialization()
    self.compute_counts()
    self.normalize_and_log()

  def get_params(self):
    return self.params

  def get_priors(self):
    return self.priors

  def cluster_parameters_initialization(self):
    """
    For each class, initializes the state transition matrix and the initial state distribution and stores them in the params dictionary.
    """
    label_2_class = self.vocabulary["label_2_class"]
    V = self.v_size
    for label in label_2_class.keys():
      self.params[label] = (np.ones((V,V)),np.ones(V))

  def compute_prior_distributions(self):
    """
    For each class, it takes the dataset's labels and computes the prior probabilities.
    """
    label_2_class = self.vocabulary["label_2_class"]
    labels = self.data['labels']
    total = len(labels)
    for k in label_2_class.keys():
      count_k = sum(y == k for y in labels)
      p_k = count_k / total
      p_k = np.log(p_k)
      self.priors[k] = p_k

  def count_for_cluster(self,dataset,A,pi):
    """
    Arguments:
      - dataset: A list of tokenized samples
      - A: The state transition matrix for class i
      - pi: The initial state distribution for class i
    i is a generic class. What I am emphasizing is that the params pairs must correspond to the same cluster
    """
    for line in dataset:
      last_token = None # we this to know whether we need to update the state transicion matrix or the initial state distribution
      for token in line:
        if last_token is None: # we are at the start of a sentence
          pi[token] += 1
        else: # we are not at the start of a sentence, so we add a count to the transition between states last_token -> token
          A[last_token,token] += 1
        # update the last token
        last_token = token


  def compute_counts(self):
    """
    Fills out the initial state distribution and the state transition matrix for each class/cluster
    """
    labels,tokenized = self.data['labels'],self.data['tokenized']
    label_2_class = self.vocabulary["label_2_class"]

    for cluster in label_2_class.keys():
      #subset = train_dataframe[train_dataframe['labels'] == cluster]['tokenized'].to_list()
      subset = [token for token,label in zip(tokenized,labels) if label == cluster] # get the subset of tokens which belong to the class in question
      self.count_for_cluster(
          dataset=subset,
          A = self.params[cluster][0],
          pi = self.params[cluster][1]
      )

  def normalize_and_log(self):
    """
    For each class, normalize and apply log probability to the state transition matrix and initial state distribution
    """
    label_2_class = self.vocabulary["label_2_class"]
    for cluster in label_2_class.keys():
      # retrieve parameters for that cluster
      Ac,pic = self.params[cluster]

      # normalize probabilities
      Ac /= Ac.sum(axis=1, keepdims=True)
      pic /= pic.sum()

      # apply log probability
      Ac = np.log(Ac)
      pic = np.log(pic)

      # update parameters
      self.params[cluster] = (Ac,pic)

  def compute_log_likelihood(self,dataset,c):

    log_Ac,log_pic = self.params[c][0],self.params[c][1]

    last_idx = None
    logprob = 0
    for idx in dataset:
      if last_idx is None:
        # it's the first token
        logprob += log_pic[idx]
      else:
        logprob += log_Ac[last_idx, idx]

      # update last_idx
      last_idx = idx

    return logprob

  def predict(self, dataset):
    K = self.k_size
    predictions = np.zeros(len(dataset))
    for i, input_ in enumerate(dataset):
      posteriors = [self.compute_log_likelihood(input_, c) + self.priors[c] \
             for c in range(K)]
      pred = np.argmax(posteriors)
      predictions[i] = pred
    return predictions
