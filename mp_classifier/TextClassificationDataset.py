from pathlib import Path
from sklearn.model_selection import train_test_split
import string
from tqdm import tqdm

class TextClassificationDataset:
  def __init__(self,data_path,test_size=0.2,verbose = False):
    """
    Receives a path with the data samples.
    """
    # class attributes
    self.data_path = Path(str(data_path))
    self.test_size = test_size
    self.verbose = verbose
    # all of these are initialized allong the preprocessing process
    self.sample_files = None
    self.train_data = None
    self.test_data = None
    self.label_2_class = None
    self.word_2_idx = {'<unk>': 0}
    self.idx_2_word = None

    # preprocess the data
    self.get_data_files()
    self.preprocess_data()
    self.create_volcabulary()
    self.tokenize_data()

    # useful functions
    self.tokenize_sentence = lambda sentence: [self.word_2_idx.get(word,0) for word in sentence.split()]
    self.detokenize_sentence = lambda sentence: ' '.join([self.idx_2_word.get(token,0) for token in sentence])


  def get_data_files(self):
    """
    extracts a file list from the provided datapath
    """
    self.sample_files = [f for f in sample_path.glob("*") if f.is_file()]
    if self.verbose:
      print(f"[+] found {len(self.sample_files)} files")


  def normalize_sentence(self,sentence):
    """
    Takes up a string and noramlzies it by removing unnecesary spaces, transforming it to lower, case, and removing puntuation signs.
    """
    sentence = sentence.lower() # take the sentence to lower case
    sentence = sentence.translate(str.maketrans("","",string.punctuation))
    return sentence

  def preprocess_data(self):
    """
    This function iterates through the list of input files and:
    - reads the file
    - preprocesses each line and assigns it a label
    - make the train test split
    -  instance the training data, test data, and text 2 label mapping as class attributes
    """

    # collect the data into lists
    input_texts,labels = list(),list()
    label_2_class = dict()
    for label,fl in enumerate(self.sample_files):
      file_name = str(fl).split("/")[-1]
      print(f"[~] parsing {file_name}")
      label_2_class[label] = file_name
      for line in tqdm(open(fl)):
        line = line.rstrip()
        if line:
          line = self.normalize_sentence(line)
          input_texts.append(line)
          labels.append(label)

    if self.verbose:
      print(f"[+] number of sentences {len(input_texts)}")
      print(f"[+] number of labels {len(labels)}")
      print(f"[+] class to label mapping:")
      print(self.label_2_class)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(input_texts,labels,test_size=self.test_size)
    # instance class attrbutes
    self.train_data = {
        "samples":X_train,
        "labels":y_train
    }
    self.test_data = {
        "samples":X_test,
        "labels":y_test
    }
    self.label_2_class = label_2_class

  def create_volcabulary(self):
    """
    Loops through the training dataset and populates the dictionary
    """

    print(f"[~] creating vocabulary...")
    idx = 1
    # create the vocabulary
    for text in tqdm(self.train_data["samples"]):
      tokens = text.split()
      for token in tokens:
        if token not in self.word_2_idx:
          self.word_2_idx[token] = idx
          idx +=1

    # create the reverse volcabulary
    self.idx_2_word = {v:k for k,v in self.word_2_idx.items()}

  def tokenize_data(self):
    """
    returns tokenized versions of the training and testing samples according to the vocabulary
    """
    tokenized_train = list()
    tokenized_test = list()
    print("[~] tokenizing training data...")
    for text in tqdm(self.train_data["samples"]):
      tokens = text.split()
      tokenized_line = [self.word_2_idx[token] for token in tokens]
      tokenized_train.append(tokenized_line)
    print("[~] tokenizing test data...")
    for text in tqdm(self.test_data["samples"]):
      tokens = text.split()
      tokenized_line = [self.word_2_idx.get(token,0) for token in tokens]
      tokenized_test.append(tokenized_line)

    self.train_data["tokenized"] = tokenized_train
    self.test_data["tokenized"] = tokenized_test

  def get_data(self):
    return self.train_data,self.test_data

  def get_vocabulary(self):
    return {
        "vocab": self.word_2_idx,
        "reverse":self.idx_2_word,
        "label_2_class":self.label_2_class
    }