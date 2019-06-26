'''
Created on 06.06.2018

Dataset base class (define template and abstract methods)
'''
from abc import ABC
from abc import abstractmethod
from abc import abstractproperty
from collections import namedtuple
import os
import sys
import time
import gzip
import pickle
import importlib

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from dataset.data_loader import Data_Loader as DL
#from data_loader import Data_Loader as DL # only for internal tests


def clock(func):
  ''' annotation for time measurement (only args not kwargs) '''

  def clocked(*args):
    t0      = time.perf_counter()
    result  = func(*args)
    elapsed = time.perf_counter() - t0
    arg_str = ', '.join(repr(arg) for arg in args)
    print('{:4.2f}s {}({}) -> {}'.format(elapsed, func.__name__, arg_str, type(result)))
    return result

  return clocked


class Dataset(ABC):
  '''
  Abstract Base Class for all datasets (template pattern)

  define:
      abstract properties (setter and getter): name, shape, pickle_file, num_train, num_test, num_classes
                                               data_train, labels_train, data_test, labels_test
      abstract methods for derived classes: download_data, prepare_data, extract_data
      template for dataset construction: get_data
  '''

  DATASET_DIR  = './datasets/'
  DOWNLOAD_DIR = DATASET_DIR + 'download/'
  
  class VALUE_RANGE():
    @property
    def min(self): return 0.0
     
    @property
    def max(self): return +1.0
     
  Range               = namedtuple('Range', 'min max')
  VALUE_RANGE         = VALUE_RANGE() # Range(0.0, +1.0) # range of data values mapped e.g. from [0, 255] to [-1.0, +1.0] 
  SPLIT_TRAINING_TEST = 10            # % if only training data exists, split training to: 90% training, 10% testing

  def __init__(self):
    ''' define internal data structure for mapping  '''
    self._data_train     = None
    self._labels_train   = None
    self._data_test      = None
    self._labels_test    = None
    self._validate_shape = True

  def __rep__(self):
    ''' string representation of a dataset '''
    s = ''
    s += 'Dataset: {}'.format(self.name)
    s += '  * pickle_file: {}'.format(self.pickle_file)
    s += '  * shape: {}'.format(self.shape)
    s += '  * num_train: {}'.format(self.num_train)
    s += '  * num_test: {}'.format(self.num_test)
    s += '  * num_classes: {}'.format(self.num_classes)

  #===============================================================================
  # fields
  #===============================================================================
  @abstractproperty
  def name(self): pass

  @abstractproperty
  def shape(self): pass

  @abstractproperty
  def pickle_file(self): pass

  @abstractproperty
  def num_train(self): pass

  @abstractproperty
  def num_test(self): pass

  @abstractproperty
  def num_classes(self): pass

  #===============================================================================
  # setter and getter
  #===============================================================================
  @property
  def validate_shape(self): return self._validate_shape

  @validate_shape.setter
  def validate_shape(self, value): self._validate_shape = value

  #-------------------------------------------------------------------------------------------------------- data_train
  @property
  def data_train(self): return self._data_train

  @data_train.setter
  def data_train(self, value): self._data_train = value

  #------------------------------------------------------------------------------------------------------ labels_train
  @property
  def labels_train(self): return self._labels_train

  @labels_train.setter
  def labels_train(self, value): self._labels_train = value

  #--------------------------------------------------------------------------------------------------------- data_test
  @property
  def data_test(self): return self._data_test

  @data_test.setter
  def data_test(self, value): self._data_test = value

  #------------------------------------------------------------------------------------------------------- labels_test
  @property
  def labels_test(self): return self._labels_test

  @labels_test.setter
  def labels_test(self, value): self._labels_test = value

  #----------------------------------------------------------------------------------------------------- is_regression
  @property
  def is_regression(self): return False
  
  #---------------------------------------------------------------------------------------------------- properties end

  def get_data(self, force=False, clean_up=False):
    ''' We'll fire you if you override this method.

    template method define the general procedure to create a dataset:
        * if: no stored data exists:
            * download data (by dataset)
            * extract data (by dataset) TF_DataSet
            * prepare data (by dataset)
            * convert: split (optional), shuffle, flatten (optional)
            * store data (pickel)
            * load data (pickel)
            * validate data loaded data
            * cleanup (optional) (by dataset)
        * else:
            * load stored data (pickle file)

    @param force: flag (boolean) for download and extract mechanism to overwrite existing files    
    
    @return: data_train (np.array), labels_train (np.array), data_test (np.array), _labels_test ((np.array), properties (dict())
    '''
    print('get_data for {}'.format(self.name))

    if Dataset.__file_exists(self.pickle_file) and not force:
      return DL.load_pickle_file(self.pickle_file)

    self.download_data(force)
    self.extract_data(force)
    self.prepare_data() # prepare_data: format, normalize, shuffle, split, pickle 
    
    if clean_up: self.clean_up()

  @abstractmethod
  def download_data(self): pass

  @abstractmethod
  def prepare_data(self): pass

  @abstractmethod
  def extract_data(self): pass

  @abstractmethod
  def clean_up(self): pass
  
  @classmethod
  def __file_exists(cls, file_or_dir):
    ''' check if file_or_dir or directory in DATASET_DIR exists

    @param file_or_dir: file or directory (str)
    
    @return: file or directory exists (bool)
     '''
    path_and_filename = os.path.join(cls.DATASET_DIR, file_or_dir)

    return os.path.isfile(path_and_filename) or os.path.isdir(path_and_filename)

def get_next_block(FLAGS=None):
  '''Load the dataset

  If pickle file exists, load it.
  Else invoke the corresponding class file (based on 'file'), create instance and return .get_data().
      The dataset will be downloaded, converted, pickled and returned (should be prepared before running experiments!).

  @param FLAGS: object (Namespace) with parameters form argparser
  
  @return: data train (np.array), labels train (np.array), data test (np.array), labels test (np.array), properties (dict(str, obj))
  '''
  if FLAGS != None:
    directory = FLAGS.dataset_dir
    file      = FLAGS.dataset_file

  pickle_file = os.path.join(directory, file)
  
  if not pickle_file.endswith('.pkl.gz'): pickle_file += '.pkl.gz'
  
  if os.path.isfile(pickle_file):
    print('\t * try load dataset: {} from {}'.format(file, directory))
    
    with gzip.open(pickle_file, 'rb') as file:
      num_elements = 0
      while True:
        try:
          save = pickle.load(file)
          yield save
        except:
          print('number of elements {}'.format(num_elements))
          return save
    
    raise Exception('file not found {}'.format(pickle_file))

  # remove file ending
  if file.endswith('.pkl.gz'): class_name = file.replace('.pkl.gz', '')
  else: class_name = file

  # import class.file (class file must have the same name as the pickle file!)
  module = importlib.import_module('dataset.' + class_name.lower())

  # load class, create instance, return get_data
  return getattr(module, class_name)().get_data(clean_up=True, force=False)

