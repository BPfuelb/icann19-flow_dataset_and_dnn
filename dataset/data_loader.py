'''
Created on 06.06.2018

Provide functions to load data from disk. 
'''
import csv
import gzip
import os
import pathlib
import pickle
import scipy.ndimage
import tarfile
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_labels
from tensorflow.python.platform import gfile
import zipfile

import numpy as np


class Data_Loader(object):
  ''' Data loader class '''
  
  DATASET_DIR  = './datasets/'
  DOWNLOAD_DIR = DATASET_DIR + 'download/'

  def __init__(self, params): pass
  
  @classmethod
  def csv_loader(cls, file):
    ''' load csv file '''
    try:
      with open(file, 'r') as csvfile:
        return np.array([[row] for row in csv.reader(csvfile)]).astype(np.float32)
    except Exception as ex:
      print(ex)


  @classmethod
  def read_idx(cls, filename):
    ''' source: https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40#file-mnist-py '''
    import struct
    with open(filename, 'rb') as f:
      _, _, dims = struct.unpack('>HBB', f.read(4))
      shape      = tuple(struct.unpack('>I', f.read(4)) [0] for _ in range(dims))
      return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

  @classmethod
  def reduce(cls, dataset):
    ''' reduce classes from dataset with least elements in class ''' 
    data_train   = dataset.data_train
    labels_train = dataset.labels_train
    data_test    = dataset.data_test
    labels_test  = dataset.labels_test
    classes      = dataset.num_classes

    def find_classes_with_most_elements(data, labels, max_classes, argsort=None):
      ''' filter data and labels (not one hot!) for a maximum of classes  
      
      1. if labels are in one hot format, convert to normal 
          e.g. [[0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1], 
                [0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0]] -> [1, 3, 5, 3, 1]
          
      2. search (max_classes) max_classes  
          if argsort is None (use training data to find most elements for class)
          e.g. max_classes = 2
               class 1 and 3 have the most element
               [1, 3, 5, 3, 1] -> [3, 1] 
      
      3. create filter only for selected classes 
          e.g. [1, == 3 or 1 ->  [T,
                3, == 3 or 1 ->   T,
                5, == 3 or 1 ->   F,
                3, == 3 or 1 ->   T,
                1] == 3 or 1 ->   T]
      4. filter data and labels with created filter 

      5a. if labels are NOT in one hot format: convert into one hot (with max classes) format and delete unused columns 
          e.g. [1, ->  [[0, 1, 0, 0, 0, 0], delete column 0, 2, 4 and 5 -> [[1, 0],
                3, ->   [0, 0, 0, 1, 0, 0],                                 [0, 1],
                3, ->   [0, 0, 0, 1, 0, 0],                                 [0, 1],
                1] ->   [0, 1, 0, 0, 0, 0]]                                 [1, 0]]
      5b. if labels are in one hot format: use index of classes with most elemets to delete columns 
          e.g. [[0, 1, 0, 0, 0, 0],  best classes [3, 1] -> [[1, 0], 
                [0, 0, 0, 1, 0, 0],                          [0, 1],
                [0, 0, 0, 1, 0, 0],                          [0, 1],
                [0, 1, 0, 0, 0, 0]]                          [1, 0]]
      '''
      # 1. convert one hot labels to normal
      argmax = labels
      if len(labels.shape) != 1: 
        # is not in one hot format 
        argmax = np.argmax(labels, axis=1)
      
      # 2. which classes have the most elements (maximum of max_classes)
      if argsort is None:
        argsort = np.bincount(argmax).argsort()[-max_classes:][::-1]
      
      # 3. create filter mask
      filter = np.isin(argmax, argsort)
      
      # 4. use filter and data and lables
      filtered_data   = data[filter]
      filtered_labels = labels[filter]
      
      # 5a. delete unused columns
      if len(labels.shape) == 1:
        to_one_hot                                                                            = np.zeros([filtered_labels.size, labels.max()])  
        to_one_hot[np.arange(filtered_labels.size), filtered_labels.astype(np.int32).ravel()] = 1  # convert labels to one_hot
    
        column_filtered_labels = to_one_hot[:, argsort]
      # 5b. delete unused columns
      else:
        column_filtered_labels = np.isin(filtered_labels, argsort)
      
      return filtered_data.astype(np.float32), column_filtered_labels.astype(np.float32), argsort
    
    dataset.data_train, dataset.labels_train, arg_sort = find_classes_with_most_elements(data_train, labels_train, classes)
    dataset.data_test, dataset.labels_test, _          = find_classes_with_most_elements(data_test, labels_test, classes, arg_sort)

  @classmethod
  def unzip_ubyte_gz(cls, file, one_hot=True):
    ''' extract tensorflow dataset format '''
    try:  # for images
      with gfile.Open(file, 'rb') as f:
        return extract_images(f)
    except:
      ValueError

    try:  # or labels
      with gfile.Open(file, 'rb') as f:
        return extract_labels(f, one_hot=one_hot)
    except:
      ValueError

  @classmethod
  def pickle_data(cls, filename, data_train, labels_train, data_test, labels_test, properties):
    ''' create pickle file
    @param filename                                        : output file name
    @param data_train, labels_train, data_test, test_lables:
    @param properties                                      : {'num_classes':int, 'num_of_channels':int, 'dimensions':[int,...]}
    @param __shuffle_data_and_labels                       : if True permute data before storing
    '''
    print('create pickle file {}'.format(filename))
    folder_filename = os.path.join(cls.DATASET_DIR, filename)
    
    
    for i in range(len(data_train)):
      save = {
          'data_train'  : data_train[i],
          'labels_train': labels_train[i],
          'data_test'   : data_test[i],
          'labels_test' : labels_test[i],
          'properties'  : properties,
      }
    
      with gzip.GzipFile(folder_filename, 'ab') as file:
        pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)

    return folder_filename
  
  @classmethod
  def load_from_pickles(cls, files, pickle_batch_size, image_size, num_of_channels):
    ''' load data and labels from file list and merge to one array 

    @param files            : list of file names (with sub directory) [str]
    @param pickle_batch_size: define number of data elements in in one file
    @param image_size       : one dimension of the image
    @param num_of_channels  : number of channels
    @return: np.array(data), np.array(labels) 
    '''

    def __load_pickle_file(pickle_file):
      ''' load data and labels from dictionary in pickle file '''
      with open(pickle_file, 'rb') as file:
        dictionary = pickle.load(file, encoding='latin1')

      return dictionary['data'], dictionary['labels']

    num_files = len(files)

    # allocate numpy arrays for data and labels
    data_arr   = np.ndarray(shape=(pickle_batch_size * num_files, image_size * image_size * num_of_channels), dtype=np.float32)
    labels_arr = np.ndarray(shape=pickle_batch_size * num_files, dtype=np.float32)
    
    # load data from each file in list and arrange it in arrays
    for index, file in enumerate(files):
      data, labels                                                            = __load_pickle_file(file)
      data_arr[index * pickle_batch_size: (index + 1) * pickle_batch_size, :] = data
      labels_arr[index * pickle_batch_size: (index + 1) * pickle_batch_size]  = labels

    return data_arr, labels_arr

  @classmethod
  def load_pickle_file(cls, filename):
    ''' load pickle filename '''

    if not filename.endswith('.pkl.gz'):
      filename += '.pkl.gz'

    folder_filename = os.path.join(cls.DATASET_DIR, filename)

    if not os.path.isfile(folder_filename):
      assert('filename {} not found'.format(filename))

    with gzip.open(folder_filename, 'rb') as file:
      save         = pickle.load(file)
      data_train   = save['data_train']
      labels_train = save['labels_train']
      data_test    = save['data_test']
      labels_test  = save['labels_test']
      properties   = save['properties']
      del save
        
    return data_train, labels_train, data_test, labels_test, properties
  
  @classmethod
  def extract(cls, file, FORCE=False):
    ''' extract a file into to target directory (create a new directory with file name)

    @param file  : path to compressed file (str)
    @param target: path to target (str) default is DOWNLOAD_DIR (a new directory will be create with same same as file)
    
    @return: path to target directory (str)
    '''

    filename = os.path.basename(file).split('.')[0]
    target   = cls.DOWNLOAD_DIR + filename

    if os.path.isdir(target) and not FORCE:
      print('target directory {} exists. do not extract'.format(file, target))
      return target

    print('extract file {} into {}'.format(file, target))
    
    if not os.path.isdir(target):
      os.makedirs(target)

    return cls.__extract_by_filetype(file, target)

  @classmethod
  def __extract_by_filetype(cls, file, target):
    ''' extract file (use file suffix)

    use tar for .tar.gz, .tgz, .pkl.gz, .tar or .gz-files
    use zip for .zip-files
    '''
    extension = ''.join(pathlib.Path(file).suffixes)

    print('file {} extension {} target {}'.format(file, extension, target))

    if extension in ['.tar.gz', '.tgz', '.pkl.gz', '.tar', '.gz']:
      tar = tarfile.open(file)
      tar.extractall(path=target)
      return target
    elif extension == '.zip':
      with zipfile.ZipFile(file, 'r') as zip_extractor:
        zip_extractor.extractall(target)
      return target
    else:
      raise Exception('invalid file type ({} from file {})'.format(extension, file))
  
  @classmethod
  def __reshape_image(cls, image, shape, interp='bicubic'):
    '''  use scipy.misc.imresize for resizing the image

    @param image : as np.array
    @param shape : target shape as (x (int), y (int)) or percentage (int) or fraction (float)
    @param interp: "nearest", "lanczos", "bilinear", default("bicubic") or "cubic"
    
    @return: resized image as numpy array
    '''
    # if image.shape != shape:
    resize_image = skimage.transform.resize(image, shape[:2], interp=interp)
    return resize_image
  
  @classmethod
  def load_directory_as_dataset(cls, training_folder, shape, test_folder=None, reshape=False, max_classes=10):
    ''' convert images to data set

    <training_folder>/
            /<class directory>/
                              /<image>
                              /<image>
                              ...
            ...
    <testing_folder>/
            /<class directory>/
                              /<image>
                              /<image>
                              ...
            ...
    @param training_folder: directory path
    @param shape          : target image shape (x, y) tupel((int), (int))
    @param test_folder    : if not None, read test images from given directory (same classes!)
            if load_directory_as_dataset is called twice (for loading training and test data) 
            the labels of traning and test data could distinguish 
    @param reshape        : if True (bool), reshape image with param shape
    
    @return: data (np.array), labels (np.array) or data_train, lables_train, data_test (np.array), labels_test (np.array) 
    '''
    print('convert images {}'.format(training_folder))
    
    # build dict with classnames (directories) as keys and list of files as values 
    classes_and_files = {class_directory: [os.path.join(training_folder, class_directory, image) 
                                           for image in os.listdir(training_folder + class_directory)] 
                                           for class_directory in os.listdir(training_folder) 
                                           if len(os.listdir(training_folder + class_directory)) >= 1}

    # sort dict -> [(k,v)] (descending)
    sorted_classes_and_files = sorted(classes_and_files.items(), key=lambda kv: len(kv[1]), reverse=True)
    
    num_of_classes = len(classes_and_files)

    data          = list()
    labels        = list()
    for_test_data = dict()
    for label, (dir, files) in enumerate(sorted_classes_and_files):

      if label == max_classes:
        print('stop at class {}/{}'.format(max_classes, num_of_classes))
        break
      
      print('class:', label, dir, 'num files:', len(files))
      
      # store label and directory name for create test data
      for_test_data[label] = dir
      
      for file in files:
        image_as_array = cls.load_image_as_array(file) # load images as array
        if image_as_array is None: continue # skip invalid images

        if reshape: # reshape image?
          image_as_array = cls.__reshape_image(image_as_array, shape)
          if shape != image_as_array.shape:
            image_as_array = np.stack((image_as_array,) * shape[2], -1)
        
        # add data and label
        data.append(image_as_array)
        labels.append(label + 1)                
  
    if test_folder is not None:
      data_test   = []
      labels_test = []
      
      for label, directory in for_test_data.items():
        print('load test data for class:', label, directory)

        for file in os.listdir(test_folder + directory):
          image_as_array = cls.load_image_as_array(test_folder + directory + '/' + file) # load images as array
          if image_as_array is None: continue                                            # skip invalid images
          if reshape:                                                                    # reshape image?
            image_as_array = cls.__reshape_image(image_as_array, shape)
            if shape != image_as_array.shape:
              image_as_array = np.stack((image_as_array,) * shape[2], -1)
          
          # add data and label
          data_test.append(image_as_array)
          labels_test.append(label)
        
        return np.array(data), np.array(labels), np.array(data_test), np.array(labels_test)
    
    data = np.array(data)
    labels = np.array(labels)
    return data, labels
  
  @classmethod
  def load_image_as_array(cls, file):
    ''' load image as numpy array '''
    try:
      data = scipy.ndimage.imread(file)
    except Exception as e:
      print('file ({}) is not an valid image ...'.format(file, e))
      return

    return data
