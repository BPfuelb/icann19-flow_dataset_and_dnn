'''
Created on 06.06.2018

HS-Flows dataset class (with additional definitions)
'''
import sys
import os
import gzip
import pickle
import psutil
import time

from dataset.dataset import Dataset as DS
# from dataset import Dataset as DS # only for local use
from functools import partial
from scipy.interpolate import interp1d
from datetime import datetime
from ipaddress import IPv4Address
import numpy as np
import collections
import math
from builtins import enumerate
from enum import Enum
import safe_cast

''' raw data from anonymizer '''
# DATASET = '2018-12-19_11-34-48_flows.pkl.gz' # from RZ
# DATASET = '2018-12-20_18-02-58_flows.pkl.gz' # from RZ
# DATASET = '2018-12-23_17-43-43_flows.pkl.gz' # own
# DATASET = '2018-12-26_15-50-12_flows.pkl.gz' # own
# DATASET = '2019-01-12_18-13-03_flows.pkl.gz' # own
# DATASET = '2019-01-18_20-36-58_flows.pkl.gz'
# DATASET = '2019-02-08_17-00-45_flows.pkl.gz'
DATASET = '2019-02-16_04-39-05_flows.pkl.gz'
# DATASET = '2019-01-31_19-31-09_flows.pkl.gz'


def timing(method):
  ''' decorator for time (and memory) measuring of functions '''

  def measure(*args, **kw):
    print('{}'.format(method.__name__), end='', flush=True)
    start  = time.time()
    result = method(*args, **kw)
    if hasattr(psutil.Process(), 'memory_info'):
      mem = psutil.Process(os.getpid()).memory_info()[0] // (2 ** 20)
      print('...{:2.2f}s mem: {}MB'.format(time.time() - start, mem))
    elif hasattr(psutil.Process(), 'memory_full_info'):
      mem = psutil.Process(os.getpid()).memory_full_info()[0] // (2 ** 20)
      print('...{:2.2f}s mem: {}MB'.format(time.time() - start, mem))
    else:
      print('...{:2.2f}s '.format(time.time() - start))
    return result

  return measure


class Feature():
  ''' enumeration class defines positions of features '''
  FS_YEAR           =  0
  FS_MONTH          =  1
  FS_DAY            =  2
  FS_HOUR           =  3
  FS_MINUTE         =  4
  FS_SECOND         =  5
  DURATION          =  6
  BYTES             =  7
  BPS               =  8
  PROTOCOL          =  9
  SRC_ADDR          = 10
  DST_ADDR          = 11
  SRC_NETWORK       = 12
  DST_NETWORK       = 13
  SRC_PORT          = 14
  DST_PORT          = 15
  SRC_PREFIX_LEN    = 16
  DST_PREFIX_LEN    = 17
  SRC_ASN           = 18
  DST_ASN           = 19
  SRC_LONGITUDE     = 20
  DST_LONGITUDE     = 21
  SRC_LATITUDE      = 22
  DST_LATITUDE      = 23
  SRC_COUNTRY_CODE  = 24
  DST_COUNTRY_CODE  = 25
  SRC_VLAN          = 26
  DST_VLAN          = 27
  SRC_LOCALITY      = 28
  DST_LOCALITY      = 29
  TCP_FLAGS         = 30
  FIRST_SWITCHED_TS = 31
  LAST_SWITCHED_TS  = 32
  FLOW_SEQ_NUM      = 33
  EXPORT_HOST       = 34
  # features only exists if correlation is done
  REV_BYTES         = 35
  REV_DURATION      = 36
  REV_BPS           = 37


FIVE_TUPLE = [
  Feature.PROTOCOL,
  Feature.SRC_ADDR,
  Feature.DST_ADDR,
  Feature.SRC_PORT,
  Feature.DST_PORT,
  ]

''' define features for removing duplicate values 
    e.g.: Feature.PROTOCOL, Feature.SRC_ADDR, Feature.DST_ADDR, Feature.SRC_PORT, Feature.DST_PORT,
      first_switched= 9:00, Protocol=17, SRC_ADDR=192.168.76.1, DST_ADDR=192.168.76.2 SRC_PORT=123, DST_PORT=54687
      first_switched=21:00, Protocol=17, SRC_ADDR=192.168.76.1, DST_ADDR=192.168.76.2 SRC_PORT=123, DST_PORT=54687
    -> duplicate entries, only the first is keept
'''
default_filter_feature_list = [ #
  # Feature.PROTOCOL,
  # Feature.SRC_ADDR, Feature.DST_ADDR,
  # Feature.SRC_NETWORK, Feature.DST_NETWORK,
  # Feature.SRC_PORT, Feature.DST_PORT,
  # Feature.SRC_PREFIX_LEN, Feature.DST_PREFIX_LEN,
  # Feature.SRC_VLAN, Feature.DST_VLAN,
  # Feature.SRC_LOCALITY, Feature.DST_LOCALITY,
  ] # no filtering

''' defines the timeout interval (between equal datagrams) for the aggregation of UDP '''
UDP_TIMEOUT            = 30000   # ms
                       
AGGREGATION_TIMEOUT    = 5000    # ms
NUM_BLOCKS             = 10      # number of blocks to load from file (each block 10.000 elements)
PORT_MAX_LIMIT         = 2 ** 15 # -1 -> off

country_abbreviations  = None # placeholder for country abbreviation dictionary

''' enable or disable aggregation methods '''
AGGREGATION_SAMPLES    = True
AGGREGATION_DIRECTIONS = False


class Datatype(Enum):
  ''' defines datatype conversion methods '''
  FLOAT   = 1
  ONE_HOT = 2
  BITS    = 3


class TCPFlags():
  ''' defines general TCP flags (used in aggregation) '''
  NONE =  0
  FIN  =  1
  SYN  =  2
  RST  =  4
  PSH  =  8
  ACK  = 16
  URG  = 32


class Timestamp():
  ''' value positions in timestamp '''
  YEAR   = 0
  MONTH  = 1
  DAY    = 2
  HOUR   = 3
  MINUTE = 4
  SECOND = 5


FEATURE_LIST      = collections.OrderedDict() # order, position and size of features (for feature slicing)
FEATURE_DENORM_FN = dict()                   # function list of inverse functions of the normalization process


def denorm_value(data_type, data, **kwargs):
  ''' function that inverse a normalization of a value (is partially bound and added to the dataset) 
    e.g. revert a already performed normalization of a value (e.g. port=0.053) back to its original value (port=3500) 
    
  @param data_type: defines data type conversion method (Datatype) (e.g. BITS, FLOAT, ONE-HOT)
  @param data     : values (column) that get converted (np.array)
  @param kwargs   : 
    'start' and 'end' defines position of a feature within the dataset
    'input_type' defines the raw data input type (e.g., np.uint32)
    'min_' and 'max_' defines the range of an value (e.g., [0,59] for miniutes)
    'min_range' and 'max_range' definesthe range of the output values (e.g., [0., 1.])
    'output_type' defines output type of values (e.g., np.uint8, extra IPv4Address)
    
  @return np.array(data)
  '''
  start = kwargs.get('start')
  end   = kwargs.get('end')
  data  = data[:, start:end]

  # region ----------------------------------------------------------------------------------------------- Datatype.BITS
  if data_type == Datatype.BITS:
    input_type = kwargs.get('input_type')
    min_       = kwargs.get('min_')
    max_       = kwargs.get('max_')

    num_values = len(range(int(min_), int(max_) + 1))
    slice_     = math.ceil(math.log(num_values, 2))

    bits               = np.iinfo(input_type).bits
    data[data == min_] = 0

    if slice_ < bits:
      data_                     = np.zeros((data.shape[0], bits)).astype(np.float32)
      data_[:, bits - slice_ :] = data
      data                      = data_

    data = data.astype(input_type)
    data = data.reshape(-1, 8)

    # reverse sequence of bytes for np.uint16 and np.uint32
    if input_type == np.uint16: 
      data[1::2], data[0::2] = data[::2].copy(), data[1::2].copy()
    if input_type == np.uint32:
      data[3::4], data[2::4], data[1::4], data[::4] = data[::4].copy(), data[1::4].copy(), data[2::4].copy(), data[3::4].copy()

    data = data.reshape(-1, bits)
    data = np.packbits(data.astype(input_type), axis=1)

    data = data.view(input_type)
  # endregion
  # region ---------------------------------------------------------------------------------------------- Datatype.FLOAT
  if data_type == Datatype.FLOAT:
    min_        = kwargs['min_']
    max_        = kwargs['max_']
    min_range   = kwargs['min_range']
    max_range   = kwargs['max_range']
    output_type = kwargs['output_type']

    if output_type == 'IPv4Address':
      data_fn = np.vectorize(interp1d([min_range, max_range], [0, np.iinfo(np.uint8).max]))
      data    = data_fn(data[:, 0:4])
      data    = data.astype(np.uint8)
      data    = np.fliplr(data)
      data    = data.copy()
      data    = data.view(dtype=np.uint32)
      data    = np.vectorize(IPv4Address, otypes=[IPv4Address])(data)

      return data

    data_fn = np.vectorize(interp1d([min_range, max_range], [min_, max_]))
    data    = data_fn(data)
    data    = data.astype(output_type)
  # endregion
  # region -------------------------------------------------------------------------------------------- Datatype.ONE_HOT
  if data_type == Datatype.ONE_HOT:
    min_ = kwargs['min_']

    data = np.fliplr(data)
    data = np.argmax(data, axis=1) + min_
  # endregion
  return data


def norm_value(data_type, data, feature_key, **kwargs):
  ''' normalization function (additionally create an inverse function of itself) 
    e.g. performed normalization of a value, e.g., port=3500 to port=0.053
     
    @param data_type  : defines data type conversion method (Datatype) (e.g. BITS, FLOAT, ONE-HOT)
    @param data       : values (column) that get converted (np.array)
    @param feature_key: position (column) of feature in data (Feature (int)) 
    @param kwargs     : 
    'input_type' defines the raw data input type (e.g., np.uint32)
    'min_' and 'max_' defines the range of an value (e.g., [0,59] for miniutes)
    'min_range' and 'max_range' definesthe range of the output values (e.g., [0., 1.])
    'output_type' defines output type of values (e.g., np.uint8, extra IPv4Address)
    
    @return np.array(data)
  '''
  # region ----------------------------------------------------------------------------------------------- Datatype.BITS
  if data_type == Datatype.BITS:
    input_type = kwargs.get('input_type')
    min_       = kwargs.get('min_', 0.)                       # if not set, minimum is 0
    max_       = kwargs.get('max_', np.iinfo(input_type).max) # if not set, maximum is determined based on input type

    feature_size              = np.iinfo(input_type).bits     # sine of feature based on input type
    start                     = sum(FEATURE_LIST.values())    # determine start position of feature
    FEATURE_LIST[feature_key] = feature_size            
    end                       = start + feature_size          # calculate end position of feature
                              
    bits                      = np.iinfo(input_type).bits     # get number of used bits for input type
    data                      = data.astype(input_type)       # convert input data to input data type
    data                      = data.view(np.uint8)           # byte view of the data (needed for unpackbits)
    data                      = np.unpackbits(data, axis=0)   # convert byte to single bits
    data                      = data.reshape(-1, 8)           # reshape back to numpy bytes array e.g., [0 1 1 0 1 0 1 0], ...

    # reverse sequence of bytes for np.uint16 and np.uint32
    if input_type == np.uint16:
      data[1::2], data[0::2] = data[::2].copy(), data[1::2].copy()
    if input_type == np.uint32:
      data[3::4], data[2::4], data[1::4], data[::4] = data[::4].copy(), data[1::4].copy(), data[2::4].copy(), data[3::4].copy()

    data = data.reshape(-1, bits)  # reshape bytes to number of input type bits
    data = data.astype(np.float32) # convert as bits as floats

    num_values = len(range(int(min_), int(max_) + 1)) # calculate number of bits with min_ and max_ values
    slice_     = math.ceil(math.log(num_values, 2))   # calculate number of bits with num_values
    if slice_ < bits:                                 # if a slice_ is smaller num bits (e.g., for VLAN only 12 bits needed)
      FEATURE_LIST[feature_key] = slice_
      data = data[:, -slice_:]
      end  = start + slice_

    # use parameters to bind partial denorm function and add to FEATURE_DENORM_FN for dataset
    FEATURE_DENORM_FN[feature_key] = partial(
      denorm_value, data_type,
      start      = start,
      end        = end,
      input_type = input_type,
      min_       = min_,
      max_       = max_,
      )

    data[data == 0.0] = min_
  # endregion
  # region ---------------------------------------------------------------------------------------------- Datatype.FLOAT
  if data_type == Datatype.FLOAT:
    min_                      = kwargs.get('min_', 0.0)
    input_type                = kwargs.get('input_type')
    max_                      = kwargs['max_'] if 'max_' in kwargs else np.iinfo(input_type).max
    min_range                 = kwargs.get('min_range')
    max_range                 = kwargs.get('max_range')
    output_type               = kwargs.get('output_type')

    feature_size              = 1
    start                     = sum(FEATURE_LIST.values())
    FEATURE_LIST[feature_key] = feature_size
    end                       = start + feature_size

    FEATURE_DENORM_FN[feature_key] = partial(
      denorm_value, data_type,
      start       = start,
      end         = end,
      min_        = min_,
      max_        = max_,
      min_range   = min_range,
      max_range   = max_range,
      output_type = output_type,
      )

    # region ------------------------------------------------------------------------------------------------ convert IP
    # e.g., A.B.C.D -> float(A), float(B), float(C), float(D)
    if output_type == np.uint32:                           # only used for IP addresses
      FEATURE_LIST[feature_key] = 4                        # add 4 float values
      data                      = data.astype(output_type) # convert to output type
      split_oct_type            = np.dtype((np.int32, {    # define split data type
        'oct0':(np.uint8, 3),
        'oct1':(np.uint8, 2),
        'oct2':(np.uint8, 1),
        'oct3':(np.uint8, 0),
        }))
      data = data.view(dtype=split_oct_type) # apply split data type

      # create and apply octet convert function on each octet
      data_fn = np.vectorize(interp1d([0, np.iinfo(np.uint8).max], [min_range, max_range]))
      oct0    = data_fn(data['oct0'])
      oct1    = data_fn(data['oct1'])
      oct2    = data_fn(data['oct2'])
      oct3    = data_fn(data['oct3'])

      data    = np.array([oct0, oct1, oct2, oct3]).T # recombine converted octets
    # endregion
    data_fn = np.vectorize(interp1d([min_, max_], [min_range, max_range]))
    data    = data_fn(data)
  # endregion
  # region -------------------------------------------------------------------------------------------- Datatype.ONE_HOT
  if data_type == Datatype.ONE_HOT:
    min_range                      = kwargs.get('min_range')
    max_range                      = kwargs.get('max_range')
    min_                           = kwargs.get('min_')
    max_                           = kwargs.get('max_')
                                   
    feature_size                   = len(range(int(min_), int(max_ + 1))) # calculate the number of one hot values
    start                          = sum(FEATURE_LIST.values())
    FEATURE_LIST[feature_key]      = feature_size
    end                            = start + feature_size

    FEATURE_DENORM_FN[feature_key] = partial(
      denorm_value,
      data_type,
      start = start,
      end   = end,
      min_  = min_,
      )

    for_all        = np.arange(data.shape[0])
    data           = data.astype(np.int32)
    value_as_index = data - min_ # shift all values based on minimum (e.g., month -> 1 => 0; 12 => 11)

    data                          = np.full((data.shape[0], max_ - min_ + 1), min_range, dtype=np.float32) # define new one hot vector
    data[for_all, value_as_index] = max_range                                                              # set value in one hot (month = 0 = [ 1 0 ... ])
    data                          = np.fliplr(data)                                                        # flip value (month = 0 = [ ... 0 1 ])
  # endregion
  return data


class HS_Flows(DS):
  ''' HS-Flow data '''

  def __init__(self):
    super(HS_Flows, self).__init__()

    self._num_classes = 0         # calculated by class_labels()
    self._shape       = (0, 0, 0) # calculated by normalize_data()
    self._num_train   = 0         # calculated by split_train_test()
    self._num_test    = 0         # calculated by split_train_test()

  @property
  def name(self): return 'HS_Flows'

  @property
  def shape(self): return self._shape

  @property
  def pickle_file(self): return 'HS_Flows.pkl.gz'

  @property
  def num_train(self): return self._num_train # individual block size minus filter deductions (duplicate filter, aggregation or correlation)

  @property
  def num_test(self): return self._num_test

  @property
  def num_classes(self): return self._num_classes

  @property
  def is_regression(self): return False

  def download_data(self, force=False):
    ''' not used here '''
    # self.compressed_file = '../datasets/...'
    pass

  def extract_data(self, force=False):
    ''' not used here '''
    # self.extracted_files = DL.extract(self.compressed_file, FORCE=force)
    pass

  def prepare_data(self):
    ''' prepare data function invoked by dataset class 
    
       1. load data block from pickle file (with load_pickle())
       2. convert strings to value types (with format_data())
       3. apply duplicate filter if defined (with filter_columns_and_remove_duplicates())
       4. aggregate samples (with aggregate_samples())
       5. (optional) aggregate directions (with aggregate_directions())
       6. substitute random chosen ports (with remove_random_ports())
       7. normalize data (with normalize_data())
       8. shuffle data (with shuffle_data())
       9. split training and test data (with split_train_test())
      10. append block to pickle file (with pickle_data())
    repeat all steps until all data are converted
    '''

    def load_pickle(filename):
      ''' load pickle file with raw data (strings) 
      use NUM_BLOCKS to combine X raw data blocks to one dataset block
      
      @param filename: predefined filename DATASET
      @return: generator function
      '''
      if not filename.endswith('.pkl.gz'): filename += '.pkl.gz'

      print('load pickle file {}'.format(filename), end='', flush=True)

      if os.path.dirname(os.path.realpath(__file__)) not in filename:
        folder_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
      else:
        folder_filename = filename
      if not os.path.isfile(folder_filename): assert('pickle file {} not found'.format(filename))

      print('...size: {} MB\n'.format(os.path.getsize(folder_filename) // 2 ** 20))

      # if the file contain multiple objects, append all to a list
      with gzip.open(folder_filename, 'rb') as file:
        num_blocks   = 0
        num_elements = 0
        while True:
          try:
            result = []
            for _ in range(NUM_BLOCKS):
              loaded_elements = pickle.load(file)
              result         += loaded_elements
              # print('len(loaded_elements)', len(loaded_elements))
              num_elements   += len(loaded_elements)
            num_blocks += 1
            print('block number: {} num_elements {}'.format(num_blocks, num_elements))
            yield result
          except Exception:
            print('last: block number: {} num_elements {}'.format(num_blocks, num_elements))
            yield result

    @timing
    def format_data(data):
      ''' convert raw data (strings) into value types (e.g., int, float, ...)
       
      @param data: list(dict(...))
      @return: data as np.array
      '''

      def _get_elem(line, name, type_, pre_fun=None, post_fun=None, default=None):
        ''' generic convert function 
        
        @param line    : line of values (dict)
        @param name    : key of the value in line
        @param type_   : type to which the value is converted by cast (e.g., int, float, ...) 
        @param pre_fun : (optional) function which is applied before cast (e.g., convert to IPv4Address object)  
        @param post_fun: (optional) function which is applied after cast (e.g., split timestamp)

        @param default : (optional) default value if key is not in line or None or 'None' 
        '''
        try:
          element = line.get(name, default)
          if pre_fun : element = pre_fun(element)
          element = safe_cast.safe_cast(element, type_, default=default)
          if post_fun: element = post_fun(element)
        except Exception:
          if default: return default
          print('invalid element for key {}: {}'.format(name, element))
        return element

      def create_country_lookup_dict():
        ''' create a global lookup dictionary for country codes '''
        # https://www.iso.org/obp/ui/#search
        global country_abbreviations
        country_abbreviations = [
          None,
          'AF', 'AX', 'AL', 'DZ', 'AS', 'AD', 'AO', 'AI', 'AQ', 'AG', # 10 per line
          'AR', 'AM', 'AW', 'AU', 'AT', 'AZ', 'BS', 'BH', 'BD', 'BB',
          'BY', 'BE', 'BZ', 'BJ', 'BM', 'BT', 'BO', 'BQ', 'BA', 'BW',
          'BV', 'BR', 'IO', 'BN', 'BG', 'BF', 'BI', 'CV', 'KH', 'CM',
          'CA', 'KY', 'CF', 'TD', 'CL', 'CN', 'CX', 'CC', 'CO', 'KM',
          'CD', 'CG', 'CK', 'CR', 'CI', 'HR', 'CU', 'CW', 'CY', 'CZ',
          'DK', 'DJ', 'DM', 'DO', 'EC', 'EG', 'SV', 'GQ', 'ER', 'EE',
          'SZ', 'ET', 'FK', 'FO', 'FJ', 'FI', 'FR', 'GF', 'PF', 'TF',
          'GA', 'GM', 'GE', 'DE', 'GH', 'GI', 'GR', 'GL', 'GD', 'GP',
          'GU', 'GT', 'GG', 'GN', 'GW', 'GY', 'HT', 'HM', 'VA', 'HN',
          'HK', 'HU', 'IS', 'IN', 'ID', 'IR', 'IQ', 'IE', 'IM', 'IL',
          'IT', 'JM', 'JP', 'JE', 'JO', 'KZ', 'KE', 'KI', 'KP', 'KR',
          'KW', 'KG', 'LA', 'LV', 'LB', 'LS', 'LR', 'LY', 'LI', 'LT',
          'LU', 'MO', 'MK', 'MG', 'MW', 'MY', 'MV', 'ML', 'MT', 'MH',
          'MQ', 'MR', 'MU', 'YT', 'MX', 'FM', 'MD', 'MC', 'MN', 'ME',
          'MS', 'MA', 'MZ', 'MM', 'NA', 'NR', 'NP', 'NL', 'NC', 'NZ',
          'NI', 'NE', 'NG', 'NU', 'NF', 'MP', 'NO', 'OM', 'PK', 'PW',
          'PS', 'PA', 'PG', 'PY', 'PE', 'PH', 'PN', 'PL', 'PT', 'PR',
          'QA', 'RE', 'RO', 'RU', 'RW', 'BL', 'SH', 'KN', 'LC', 'MF',
          'PM', 'VC', 'WS', 'SM', 'ST', 'SA', 'SN', 'RS', 'SC', 'SL',
          'SG', 'SX', 'SK', 'SI', 'SB', 'SO', 'ZA', 'GS', 'SS', 'ES',
          'LK', 'SD', 'SR', 'SJ', 'SE', 'CH', 'SY', 'TW', 'TJ', 'TZ',
          'TH', 'TL', 'TG', 'TK', 'TO', 'TT', 'TN', 'TR', 'TM', 'TC',
          'TV', 'UG', 'UA', 'AE', 'GB', 'UM', 'US', 'UY', 'UZ', 'VU',
          'VE', 'VN', 'VG', 'VI', 'WF', 'EH', 'YE', 'ZM', 'ZW',
          ]
        dict_ = { abbrev: id_ for id_, abbrev in enumerate(country_abbreviations) }
        dict_.update({'None':0})  # None and 'None' -> 0
        return dict_

      country_lookup_dict = create_country_lookup_dict()

      def split_timestamp(timestamp_):
        ''' a timestamp split function (remove milliseconds) 
        
        @param timestamp_: timestamp to convert (with milliseconds)
        @return: converted timestamp tuple(year, month, day, hour, minute, second), original timestamp 
        '''
        timestamp = datetime.utcfromtimestamp(timestamp_ // 1000)
        timestamp = (
          timestamp.year,
          timestamp.month,
          timestamp.day,
          timestamp.hour,
          timestamp.minute,
          timestamp.second,
          )
        return timestamp, timestamp_

      def ip_addr(ip):
        ''' convert IP string to IPv4Address object
        
        @raise AddressValueError: if invalid IPv4 address
        @return: ipaddress.IPv4Address
        '''
        return IPv4Address(ip)

      def port_limit(port):
        ''' substitute ports by PORT_MAX_LIMIT
        
        @deprecated: done after aggregation in remove_random_ports() 
        @param port: value of port 
        @return: original or replaced port
        '''
        if PORT_MAX_LIMIT < 0    : return port
        if port >= PORT_MAX_LIMIT: return 0
        return port

      def country_lookup(country_code):
        ''' use country code lookup dictionary (e.g., AF -> 1)
        
        @param country_code: country code (str)
        @return: substituted country code (int) 
        '''
        return country_lookup_dict.get(country_code, 0)

      def locality(locality):
        ''' substitude locality string with int (private -> 1, public -> 0)
        
        @param locality: locality (str)
        @return: locality (int)  
        '''
        return 1 if locality == 'private' else 0

      def replace_none(vlan):
        ''' replace none value in vlans 
        
        @deprecated: is fixed in anonymizer # FIXME: 
        @param vlan: value (str)
        @return: original vlan value (int) or replaced none value
        '''
        if vlan is None  : return 0
        if vlan == None  : return 0
        if vlan == 'None': return 0
        return vlan

      dataset = []

      def print_line(l):
        ''' debug function to print a line '''
        # print(l)
        print(
          ('first_switched {} duration {:<7} bytes {:<5} bps {:<18} protocol {:<2} src_addr {:<15} dst_addr {:<15} '
          'src_network {:<15} dst_network {:<15} src_port {:<5} dst_port {:<5} '
          'src_prefix_len {:<2} dst_prefix_len {:<2} src_asn {:<5} dst_asn {:<5} '
          'src_longitude {:<9} dst_longitude {:<9} src_latitude {:<9} dst_latitude {:<9} '
          'src_country_code {} dst_country_code {} src_vlan {} dst_vlan {} src_locality {} dst_locality {} '
          'tcp_flags {} flow_seq_num {} export_host {}').format(
          l.get('first_switched', '0'),
          l.get('duration', '0'),
          l.get('bytes', '0'),
          l.get('bps', '0'),
          l.get('protocol', '0'),
          l.get('src_addr', '0'),
          l.get('dst_addr', '0'),
          l.get('src_network', '0'),
          l.get('dst_network', '0'),
          l.get('src_port', '0'),
          l.get('dst_port', '0'),
          l.get('src_prefix_len', '0'),
          l.get('dst_prefix_len', '0'),
          l.get('src_asn', '0'),
          l.get('dst_asn', '0'),
          l.get('src_longitude', '0'),
          l.get('dst_longitude', '0'),
          l.get('src_latitude', '0'),
          l.get('dst_latitude', '0'),
          l.get('src_country_code', '0'),
          l.get('dst_country_code', '0'),
          l.get('src_vlan', '0'),
          l.get('dst_vlan', '0'),
          l.get('src_locality', '0'),
          l.get('dst_locality', '0'),
          l.get('tcp_flags', '0'),
          l.get('flow_seq_num', '0'),
          l.get('host', '0'),
          ))

      for n, line in enumerate(data): # format all values in one line
        # if n == 0: print_line(line)
        # if n == len(data) - 1: print_line(line)
        # print_line(line)

        first_switched, first_switched_ts = _get_elem(line, 'first_switched', int, post_fun=split_timestamp)
        last_switched, last_switched_ts   = _get_elem(line, 'last_switched', int, post_fun=split_timestamp)
        duration                          = 0 # is calculated in aggregate_samples
        bytes_                            = _get_elem(line, 'bytes', int)
        bps                               = 0 # is calculated in aggregate_samples
        protocol                          = _get_elem(line, 'protocol', int)
        src_addr                          = _get_elem(line, 'src_addr', int, pre_fun=ip_addr)
        dst_addr                          = _get_elem(line, 'dst_addr', int, pre_fun=ip_addr)
        src_network                       = _get_elem(line, 'src_network', int, pre_fun=ip_addr)
        dst_network                       = _get_elem(line, 'dst_network', int, pre_fun=ip_addr)
        src_port                          = _get_elem(line, 'src_port', int) # port substitution is done after aggregation
        dst_port                          = _get_elem(line, 'dst_port', int) # port substitution is done after aggregation
        src_prefix_len                    = _get_elem(line, 'src_prefix_len', int)
        dst_prefix_len                    = _get_elem(line, 'dst_prefix_len', int)
        src_asn                           = _get_elem(line, 'src_asn', int, default=0)
        dst_asn                           = _get_elem(line, 'dst_asn', int, default=0)
        src_longitude                     = _get_elem(line, 'src_longitude', float, default=0.0)
        dst_longitude                     = _get_elem(line, 'dst_longitude', float, default=0.0)
        src_latitude                      = _get_elem(line, 'src_latitude', float, default=0.0)
        dst_latitude                      = _get_elem(line, 'dst_latitude', float, default=0.0)
        src_country_code                  = _get_elem(line, 'src_latitude', str, default='None', post_fun=country_lookup)
        dst_country_code                  = _get_elem(line, 'src_latitude', str, default='None', post_fun=country_lookup)
        src_vlan                          = _get_elem(line, 'src_vlan', int, default=0 , pre_fun=replace_none) # TODO: remove pre_fun (workaround)
        dst_vlan                          = _get_elem(line, 'dst_vlan', int, default=0 , pre_fun=replace_none) # TODO: remove pre_fun (workaround)
        src_locality                      = _get_elem(line, 'src_locality', str, default='public', post_fun=locality)
        dst_locality                      = _get_elem(line, 'dst_locality', str, default='public', post_fun=locality)
        tcp_flags                         = _get_elem(line, 'tcp_flags', int, default=0)
        flow_seq_num                      = _get_elem(line, 'flow_seq_num', int)
        export_host                       = _get_elem(line, 'host', int, pre_fun=ip_addr)
        # src_postal                      = _get_elem(line, 'src_postal', str)
        # dst_postal                      = _get_elem(line, 'dst_postal', str)

        sample = [ # combine all formatted samples
          first_switched[Timestamp.YEAR],
          first_switched[Timestamp.MONTH],
          first_switched[Timestamp.DAY],
          first_switched[Timestamp.HOUR],
          first_switched[Timestamp.MINUTE],
          first_switched[Timestamp.SECOND],
          duration,
          bytes_, bps,
          protocol, 
          src_addr,         dst_addr,
          src_network,      dst_network, 
          src_port,         dst_port, 
          src_prefix_len,   dst_prefix_len,
          src_asn,          dst_asn, 
          src_longitude,    dst_longitude, 
          src_latitude,     dst_latitude,
          src_country_code, dst_country_code,
          src_vlan,         dst_vlan,
          src_locality,     dst_locality,
          tcp_flags,
          first_switched_ts,
          last_switched_ts,
          flow_seq_num,
          export_host,
          ]

        dataset.append(sample) # append formatted samples as list

      return np.array(dataset, dtype=object)

    def filter_columns_and_remove_duplicates(data, feature_list=default_filter_feature_list):
      ''' create a filter and remove duplicated entries
       
      @param data: input data (np.array)
      @param feature_list: selected parameters to filter duplicated values (list(Feature))
      @return: data without duplicate values (np.array)
      '''
      if len(default_filter_feature_list) == 0: return data

      selected_features = data[:, feature_list]               # slice all features to filter
      selected_features = selected_features.astype(np.uint32) # convert types to an universal type

      _, filtered_indizes = np.unique(selected_features, return_index=True, axis=0) # compute unique indices
      data = data[np.sort(filtered_indizes)]                                        # apply unique indices

      print('number of flows after duplicate filter', len(filtered_indizes))
      return data

    @timing
    def normalize_data(data):
      ''' normalize data 
        
        @param data: all data values (np.array)
        @return: normalized data [first_switched, protocol, ip, port, pref_len, asn, geo_coordinates, country_code, 
                                  vlan, locality, tcp_flags,], class labels [duration, bps, cor_duration, cor_bps] 
                                  (np.array, np.array)
      '''
      FEATURE_LIST.clear()

      def interpolate(min_range, max_range, data):
        ''' helper function for the interpolation of float values with predefined range values
        
        @param min_range: lower limit for interpolation
        @param max_range: upper limit for interpolation
        @param data: input data 
        @return: interpolated value 
        '''
        return interp1d([min_range, max_range], [DS.VALUE_RANGE.min, DS.VALUE_RANGE.max])(data)

      def norm_first_switched(data: np.array, data_type: Datatype) -> np.array:
        ''' normalize first_switched
          
        @param data: numpy array
        @param data_type: set the return structure of data (Datatype)
          e.g. 22.6.1960 14:14:14
               Datatype.BITS    ->  0 0 0 0 1 1 1 0, 0 0 0 0 0 1 1 0, 0 0 0 0 1 1 1 0, 0 0 0 0 1 1 1 0, 0 0 0 0 1 1 1 0   
               Datatype.FLOAT   ->      0.74432    ,       0.5      ,      0.6324    ,      0.27891    ,    0.27891
               Datatype.ONE_HOT -> 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0, # d
                                ->                                       0 0 0 0 0 0 1 0 0 0 0 0, # m 
                                ->               0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0, # h
                                ->           ... 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0, # m
                                ->           ... 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0, # s 
        @return: [fs_month, fs_day, fs_hour, fs_minute, fs_second] (np.array)
        '''
        FEATURE_LIST['fs_year'] = 0

        def _norm_first_switched(data, data_type):
          fs_month = norm_value(data_type, data[:, Feature.FS_MONTH], 'fs_month',
                                min_=1., max_=12.,
                                min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                output_type=np.uint8, input_type=np.uint8,
                                )
          fs_day = norm_value(data_type, data[:, Feature.FS_DAY], 'fs_day',
                              min_=1., max_=31.,
                              min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                              output_type=np.uint8, input_type=np.uint8,
                              )
          fs_hour = norm_value(data_type, data[:, Feature.FS_HOUR], 'fs_hour',
                               min_=0., max_=23.,
                               min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                               output_type=np.uint8, input_type=np.uint8,
                               )
          fs_minute = norm_value(data_type, data[:, Feature.FS_MINUTE], 'fs_minute',
                                 min_=0., max_=59.,
                                 min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                 output_type=np.uint8, input_type=np.uint8,
                                 )
          fs_second = norm_value(data_type, data[:, Feature.FS_SECOND], 'fs_second',
                                 min_=0., max_=59.,
                                 min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                 output_type=np.uint8, input_type=np.uint8,
                                 )
          return [fs_month, fs_day, fs_hour, fs_minute, fs_second]

        first_switched = _norm_first_switched(data, data_type)

        if data_type == Datatype.FLOAT:   return np.array(first_switched).T
        if data_type == Datatype.BITS:    return np.concatenate((first_switched), axis=1)
        if data_type == Datatype.ONE_HOT: return np.concatenate((first_switched), axis=1)

      def class_labels(data: np.array) -> np.array:
        ''' create class labels from bps and duration in one hot format 
            
        @param data: numpy array
        @return: [classes] (np.array) (one hot format)
        '''
        FEATURE_LIST['duration'] = 0
        FEATURE_LIST['bytes'] = 0
        FEATURE_LIST['bps'] = 0

        if AGGREGATION_DIRECTIONS:
          label_list = [
            Feature.BYTES,
            Feature.DURATION,
            Feature.BPS,
            Feature.REV_BYTES,
            Feature.REV_DURATION,
            Feature.REV_BPS,
            ]
        else:
          label_list = [
            Feature.BYTES,
            Feature.DURATION,
            Feature.BPS,
            ]
        return data[:, label_list]

      def norm_protocol(data: np.array, data_type: Datatype) -> np.array:
        ''' convert protocol  
          
        @param data: numpy array
        @param data_type: set the return structure of data (Datatype)
          e.g. 6
               Datatype.BITS    ->  0 0 0 0 0 1 1 0 
               Datatype.FLOAT   ->      0.0004  
               Datatype.ONE_HOT -> not implemented yet 
        @return: [protocol] (np.array)
        '''
        protocol = norm_value(data_type, data[:, Feature.PROTOCOL], 'protocol',
                              min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                              output_type=np.uint8, input_type=np.uint8,
                              )

        if data_type == Datatype.BITS:    return protocol
        if data_type == Datatype.FLOAT:   return np.array([protocol]).T
        if data_type == Datatype.ONE_HOT: return protocol

      def norm_ip(data: np.array, data_type: Datatype) -> np.array:
        ''' convert src_addr, dst_addr, 
                    src_net_addr, dst_net_addr, 
          
        @param data: numpy array
        @param data_type: set the return structure of data (Datatype)
          e.g. 0.255.0.255
               Datatype.BITS    ->  0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
               Datatype.FLOAT   ->         0               1               0               1  
        @raise NotImplementedError: data_type == Datatype.ONE_HOT
        @return: [src_addr, dst_addr, src_net_addr, dst_net_addr] (np.array)
        '''

        if data_type == Datatype.ONE_HOT: raise NotImplementedError()

        src_addr = norm_value(data_type, data[:, Feature.SRC_ADDR], 'src_addr',
                              min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                              output_type=np.uint32, input_type=np.uint32
                              )
        dst_addr = norm_value(data_type, data[:, Feature.DST_ADDR], 'dst_addr',
                              min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                              output_type=np.uint32, input_type=np.uint32
                              )
        src_network = norm_value(data_type, data[:, Feature.SRC_NETWORK], 'src_network',
                                 min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                 output_type=np.uint32, input_type=np.uint32
                                 )
        dst_network = norm_value(data_type, data[:, Feature.DST_NETWORK], 'dst_network',
                                 min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                 output_type=np.uint32, input_type=np.uint32
                                 )

        return np.concatenate((src_addr, dst_addr, src_network, dst_network), axis=1)

      def norm_port(data: np.array, data_type: Datatype) -> np.array:
        ''' convert src_port, dst_port 
          
        @param data: numpy array
        @param data_type: set the return structure of data (Datatype)
          e.g. 21
               Datatype.BITS    ->  0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1
               Datatype.FLOAT   ->         0.0000021  
        @raise NotImplementedError: data_type == Datatype.ONE_HOT
        @return: [src_port, dst_port] (np.array)
        '''

        if data_type == Datatype.ONE_HOT: raise NotImplementedError()

        src_port = norm_value(data_type, data[:, Feature.SRC_PORT], 'src_port',
                              min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                              output_type=np.uint16, input_type=np.uint16
                              )
        dst_port = norm_value(data_type, data[:, Feature.DST_PORT], 'dst_port',
                              min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                              output_type=np.uint16, input_type=np.uint16
                              )

        if data_type == Datatype.BITS:  return np.concatenate((src_port, dst_port), axis=1)
        if data_type == Datatype.FLOAT: return np.array([src_port, dst_port]).T

      def norm_pref_len(data: np.array, data_type: Datatype) -> np.array:
        ''' convert src_pref_len, dst_pref_len
          
        @param data: numpy array
        @param data_type: set the return structure of data (Datatype)
          e.g. 22
               Datatype.BITS    ->  0 0 0 1 0 1 1 0
               Datatype.FLOAT   ->    0.0000024  
        @raise NotImplementedError: data_type == Datatype.ONE_HOT
        @return: [src_pref_len, dst_pref_len] (np.array)
        '''

        if data_type == Datatype.ONE_HOT: raise NotImplementedError()

        src_prefix_len = norm_value(data_type, data[:, Feature.SRC_PREFIX_LEN], 'src_prefix_len',
                                    min_=0, max_=32,
                                    min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                    output_type=np.uint8, input_type=np.uint8,
                                    )
        dst_prefix_len = norm_value(data_type, data[:, Feature.DST_PREFIX_LEN], 'dst_prefix_len',
                                    min_=0, max_=32,
                                    min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                    output_type=np.uint8, input_type=np.uint8,
                                    )

        if data_type == Datatype.BITS:  return np.concatenate((src_prefix_len, dst_prefix_len), axis=1)
        if data_type == Datatype.FLOAT: return np.array([src_prefix_len, dst_prefix_len]).T

      def norm_asn(data: np.array, data_type: Datatype) -> np.array:
        ''' convert source and destination asn 
        
        @param data: numpy array
        @param data_type: set the return structure of data (Datatype)
          e.g. 22
               Datatype.BITS    ->  0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0
               Datatype.FLOAT   ->    0.0000014  
        @raise NotImplementedError: data_type == Datatype.ONE_HOT 
        @return: [src_asn, dst_asn] (np.array)
        '''

        if data_type == Datatype.ONE_HOT: raise NotImplementedError()

        src_asn = norm_value(data_type, data[:, Feature.SRC_ASN], 'src_asn',
                             min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                             output_type=np.uint16, input_type=np.uint16,
                             )
        dst_asn = norm_value(data_type, data[:, Feature.DST_ASN], 'dst_asn',
                             min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                             output_type=np.uint16, input_type=np.uint16,
                             )

        if data_type == Datatype.BITS:  return np.concatenate((src_asn, dst_asn), axis=1)
        if data_type == Datatype.FLOAT: return np.array([src_asn, dst_asn]).T

      def norm_geo_coordinates(data: np.array, data_type: Datatype) -> np.array:
        ''' normalize longitude and latitude
          
        @param data: numpy array
        @param data_type: set the return structure of data (Datatype)
          e.g. -51.0001245
               Datatype.FLOAT   ->    0.1000014  
        @raise NotImplementedError: data_type == Datatype.BITS
        @raise NotImplementedError: data_type == Datatype.ONE_HOT 
        @return: [longitude, latitude] (np.array)
        '''

        if data_type == Datatype.BITS:    raise NotImplementedError()
        if data_type == Datatype.ONE_HOT: raise NotImplementedError()

        # https://www.mapbox.com/help/define-lat-lon/
        src_longitude = norm_value(data_type, data[:, Feature.SRC_LONGITUDE], 'src_longitude',
                                   min_=-90., max_=90.,
                                   min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                   output_type=np.float32, input_type=np.float32,
                                   )
        dst_longitude = norm_value(data_type, data[:, Feature.DST_LONGITUDE], 'dst_longitude',
                                   min_=-90., max_=90.,
                                   min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                   output_type=np.float32, input_type=np.float32,
                                   )
        src_latitude = norm_value(data_type, data[:, Feature.SRC_LATITUDE], 'src_latitude',
                                  min_=-180., max_=180.,
                                  min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                  output_type=np.float32, input_type=np.float32,
                                  )
        dst_latitude = norm_value(data_type, data[:, Feature.DST_LATITUDE], 'dst_latitude',
                                  min_=-180., max_=180.,
                                  min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                  output_type=np.float32, input_type=np.float32,
                                  )

        return np.array([src_longitude, dst_longitude, src_latitude, dst_latitude]).T

      def norm_country_code(data: np.array, data_type: Datatype) -> np.array:
        ''' normalize country code
          
        @param data: numpy array
        @param data_type: set the return structure of data (Datatype)
          e.g. 22
               Datatype.BITS    ->      0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0
               Datatype.FLOAT   ->                0.0000014  
               Datatype.ONE_HOT ->  ... 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0        
        @return: [src_country_code, dst_country_code] (np.array)
        '''

        src_country_code = norm_value(data_type, data[:, Feature.SRC_COUNTRY_CODE], 'src_country_code',
                                  min_=0, max_=len(country_abbreviations) - 1,  # -1 because None and 'None'
                                  min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                  output_type=np.uint8, input_type=np.uint8,
                                  )
        dst_country_code = norm_value(data_type, data[:, Feature.DST_COUNTRY_CODE], 'dst_country_code',
                                  min_=0, max_=len(country_abbreviations) - 1,  # -1 because None and 'None'
                                  min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                  output_type=np.uint8, input_type=np.uint8,
                                  )

        if data_type == Datatype.BITS:    return np.concatenate((src_country_code, dst_country_code), axis=1)
        if data_type == Datatype.FLOAT:   return np.array([src_country_code, dst_country_code]).T
        if data_type == Datatype.ONE_HOT: return np.concatenate((src_country_code, dst_country_code), axis=1)

      def norm_vlan(data: np.array, data_type: Datatype) -> np.array:
        ''' convert source and destination vlan
        
        @param data: numpy array
        @param data_type: set the return structure of data (Datatype)
          e.g. 22
               Datatype.BITS    ->  0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0
               Datatype.FLOAT   ->    0.0000014 
        @raise NotImplementedError: data_type == Datatype.ONE_HOT        
        @return: source and destination vlan (np.array, np.array)
        '''
        if data_type == Datatype.ONE_HOT: raise NotImplementedError()

        src_vlan = norm_value(data_type, data[:, Feature.SRC_VLAN], 'src_vlan',
                              min_=0, max_=4095,
                              min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                              output_type=np.uint16, input_type=np.uint16
                              )
        dst_vlan = norm_value(data_type, data[:, Feature.DST_VLAN], 'dst_vlan',
                              min_=0, max_=4095,
                              min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                              output_type=np.uint16, input_type=np.uint16
                              )

        if data_type == Datatype.BITS:  return np.concatenate((src_vlan, dst_vlan), axis=1)
        if data_type == Datatype.FLOAT: return np.array([src_vlan, dst_vlan]).T

      def norm_locality(data: np.array, data_type: Datatype) -> np.array:
        ''' convert source and destination locality
        
        @param data: numpy array
        @param data_type: set the return structure of data (Datatype)
          e.g. 0
               Datatype.BITS    ->  0
        @raise NotImplementedError: data_type == Datatype.ONE_HOT
        @raise NotImplementedError: data_type == Datatype.Datatype.FLOAT                        
        @return: source and destination locality (np.array, np.array)
        '''
        if data_type == Datatype.ONE_HOT: raise NotImplementedError()
        if data_type == Datatype.FLOAT:   raise NotImplementedError()

        global country_lookup_dict

        src_locality = norm_value(data_type, data[:, Feature.SRC_LOCALITY], 'src_locality',
                                  min_=0, max_=1,
                                  min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                  output_type=np.uint8, input_type=np.uint8
                                  )
        dst_locality = norm_value(data_type, data[:, Feature.DST_LOCALITY], 'dst_locality',
                                  min_=0, max_=1,
                                  min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                                  output_type=np.uint8, input_type=np.uint8
                                  )

        return np.concatenate((src_locality, dst_locality), axis=1)

      def norm_tcp_flags(data: np.array, data_type: Datatype) -> np.array:
        ''' convert tcp_flags  
          
        @param data: numpy array
        @param data_type: set the return structure of data (Datatype)
          e.g. 18
               Datatype.BITS    ->  0 0 0 1 0 0 1 0 
               Datatype.FLOAT   ->      0.07  
        @raise NotImplementedError: data_type == Datatype.ONE_HOT
        @return: [tcp_flags] (np.array)
        '''

        if data_type == Datatype.ONE_HOT: raise NotImplementedError()

        tcp_flags = norm_value(data_type, data[:, Feature.TCP_FLAGS], 'tcp_flags',
                               min_range=DS.VALUE_RANGE.min, max_range=DS.VALUE_RANGE.max,
                               output_type=np.uint8, input_type=np.uint8,
                               )

        if data_type == Datatype.BITS:  return tcp_flags
        if data_type == Datatype.FLOAT: return np.array([tcp_flags]).T

      first_switched = norm_first_switched(data, Datatype.FLOAT)
      class_ = class_labels(data)
      protocol = norm_protocol(data, Datatype.BITS)
      ip = norm_ip(data, Datatype.BITS)
      port = norm_port(data, Datatype.BITS)
      pref_len = norm_pref_len(data, Datatype.BITS)
      asn = norm_asn(data, Datatype.BITS)
      geo_coordinates = norm_geo_coordinates(data, Datatype.FLOAT)
      country_code = norm_country_code(data, Datatype.FLOAT)
      vlan = norm_vlan(data, Datatype.BITS)
      locality = norm_locality(data, Datatype.BITS)
      tcp_flags = norm_tcp_flags(data, Datatype.BITS)

      data = np.concatenate((# combine all normalized columns and labels
        first_switched,
        protocol,
        ip,
        port,
        pref_len,
        asn,
        geo_coordinates,
        country_code,
        vlan,
        locality,
        tcp_flags,
        ), axis=1).astype(np.float32), class_.astype(np.float32)

      self._shape = (data[0].shape[1], 1, 1)
      return data

    def shuffle_data(data:np.array, labels:np.array) -> (np.array, np.array):
      ''' shuffle data and labels 
  
      @param data: input data (np.array)
      @param labels: input labels (np.array)
  
      @return: shuffled data (np.array), shuffled labels (np.array)
      '''
      permutation = np.random.RandomState(seed=42).permutation(data.shape[0])
      data = data[permutation]
      labels = labels[permutation]
      return data, labels

    def split_train_test(data:np.array, labels:np.array, percent_test:int) -> (np.array, np.array, np.array, np.array):
      ''' split the input data and labels into training data and labels and test data and labels by a given percentage
  
      @param data: input data (np.array)
      @param labels: input labels (np.array)
      @param percent_test: percent value (int) of the amount of test data/lables (default = 10%)
      @return: data_train (np.array), labels_train (np.array), data_test (np.array), labels_test (np.array)
      '''
      num_elements_train = int(data.shape[0] * ((100 - percent_test) / 100))

      data_train = data[:num_elements_train]
      labels_train = labels[:num_elements_train]

      data_test = data[num_elements_train:]
      label_test = labels[num_elements_train:]

      self._num_train = len(data_train)
      self._num_test = len(data_test)

      return data_train, labels_train, data_test, label_test

    def pickle_data(filename:str, data_train:np.array, labels_train:np.array, data_test:np.array, labels_test:np.array) -> str:
      ''' create pickle file
      @param filename: output file name
      @param data_train, labels_train, data_test, test_lables:
      @param __shuffle_data_and_labels: if True permute data before storing
      '''
      folder_filename = os.path.join(self.DATASET_DIR, filename)

      properties = { # add properties dictionary for building DNN and label creation
        'num_classes': self.num_classes,
        'num_of_channels': self.shape[2],
        'dimensions': list(self.shape[0:-1]),
        'aggregation': [AGGREGATION_SAMPLES, AGGREGATION_DIRECTIONS],
        }

      save = {
        'data_train'        : data_train,
        'labels_train'      : labels_train,
        'data_test'         : data_test,
        'labels_test'       : labels_test,
        'properties'        : properties,
        'feature_list'      : FEATURE_LIST,
        'feature_denorm_fn' : FEATURE_DENORM_FN,
      }

      with gzip.GzipFile(folder_filename, 'ab') as file:
        pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)

      return folder_filename

    @timing
    def aggregate_samples(data: np.array) -> np.array:
      ''' aggregate samples 
      
      1. sort data by first_switched_ts (stable sort)
      2. slice 5-tuple () for basis of aggregation
      3. select indices of first occurrence of each unique 5-tuple
      for each unique 5-tuple (indice) 
        4. combine all flows with same 5-tuple
        5. sort the flows based on time stamp, then sequence number
        (OR) 6a. only one flow detected -> calculate duration and bit rate (BPS) and append to aggregated flows
        (OR) 6b. if flows belong to TCP (see tcp())
        (OR) 6c. if flows belong to UDP (see udp())
        (OR) 6d. neither TCP nor UDP -> append to aggregated flows
      @param data: data to aggregate (np.array)
      @return: aggregated samples (np.array)
      '''
      if not AGGREGATION_SAMPLES: return data

      # region --------------------------------------------------------- 1. sort data by first_switched_ts (stable sort)
      data = data[data[:, Feature.FIRST_SWITCHED_TS].argsort(kind='mergesort')] # sort first_switched_ts; mergesort = stable
      # endregion

      # region ------------------------------------------------------------ 2. slice 5-tuple () for basis of aggregation
      data_ = data[:, FIVE_TUPLE].astype(np.uint32) # get only 5-tuple
      # endregion

      # region -------------------------------------------- 3. select indices of first occurrence of each unique 5-tuple
      _, unique_5tuples_indices = np.unique(data_, return_index=True, axis=0) # find all unique 5-tuples
      unique_5tuples_indices = np.sort(unique_5tuples_indices)

      # endregion
      aggregated_data = []

      for i in unique_5tuples_indices: # find all 5-tuples by mask
        # region ---------------------------------------------------------------- 4. combine all flows with same 5-tuple
        mask = data_[:, ] == data_[i]
        mask = np.all(mask, axis=1)
        equal_5tuples = data[mask] # all samples for same 5-tuple
        # endregion

        # region ------------------------------------------- 5. sort the flows based on time stamp, then sequence number
        # The last key in the sequence (first switched) is used for the primary sort order,
        # the second-to-last key (flow sequence number) for the secondary sort order, and so on.
        sort_indices_5tuples = np.lexsort((equal_5tuples[:, Feature.FLOW_SEQ_NUM], equal_5tuples[:, Feature.FIRST_SWITCHED_TS]))
        equal_5tuples = equal_5tuples[sort_indices_5tuples]

        # endregion
        def udp(data_udp: np.array):
          ''' aggregate udp flow samples for unique udp-5-tuple
  
            1. calculate time distances between flows
            2. indicate flows that exceed the UDP-timeout (UDP_TIMEOUT) -> stop_indeces
            for each stop index:
              3. calculate time gaps in flows -> time
              4. calculate sum of bytes, whole duration - time and bit rate (BPS)
              5. append aggregated flow 
              
          @param data_udp: samples for unique udp-5-tuple (np.array)
          '''
          data_udp = data_udp[data_udp[:, Feature.EXPORT_HOST] == data_udp[0, Feature.EXPORT_HOST]] # select export host from first example for all examples
          # region ----------------------------------------------------------- 1. calculate time distances between flows
          time_diff = data_udp[:, [Feature.FIRST_SWITCHED_TS, Feature.LAST_SWITCHED_TS]] # get only first- and last-switched
          time_diff = time_diff.reshape(-1, 1) # reshape to one column
          time_diff = np.squeeze(time_diff)    # remove single-dimensional entries
          time_diff = np.diff(time_diff)       # calculate difference ls_n - fs_n, fs_n+1 - ls_n

          time_diff_ = time_diff[1::2]
          # endregion
          # region ----------------------------------------- 2. indicate flows that exceed the UDP-timeout (UDP_TIMEOUT)
          stop_indices = np.where(time_diff_ > UDP_TIMEOUT) # return (array, dtype)
          stop_indices = stop_indices[0]                    # select only array

          start_index = 0
          stop_indices = stop_indices.tolist()
          stop_indices.append(len(data_udp) - 1)
          # endregion
          for stop_index in stop_indices:
            # region ------------------------------------------------------------------- 3. calculate time gaps in flows
            stop_index += 1

            sum_bytes = np.sum(data_udp[start_index: stop_index, Feature.BYTES])
            max_ls = np.max(data_udp[start_index: stop_index, Feature.LAST_SWITCHED_TS])

            time_diff_slice = time_diff[start_index * 2: stop_index * 2]
            i1 = np.where(time_diff_slice < UDP_TIMEOUT)[0]
            i2 = np.where(time_diff_slice > 0)[0]
            i3 = np.intersect1d(i1, i2)
            i3 = i3[i3 % 2 == 1]
            time = np.sum(time_diff_slice[i3])
            if not time: time = 0.
            # endregion
            # region ------------------------------- 4. calculate sum of bytes, whole duration - time and bit rate (BPS)
            data_udp[start_index, Feature.BYTES] = sum_bytes
            data_udp[start_index, Feature.LAST_SWITCHED_TS] = max_ls
            duration = ((max_ls - data_udp[start_index, Feature.FIRST_SWITCHED_TS]) - time) / 1000
            data_udp[start_index, Feature.DURATION] = duration
            if duration > 0: data_udp[start_index, Feature.BPS] = 8 * sum_bytes / duration
            # endregion
            # region ------------------------------------------------------------------------- 5. append aggregated flow
            aggregated_data.append(data_udp[start_index])
            start_index = stop_index
            # endregion

        def tcp(data_tcp: np.array):
          ''' aggregate tcp flow samples for unique tcp-5-tuple
          
            1. filter flows based of first appearance on export host
            2. calculate time difference between flows -> time_diff
            3. determine connection ends based on TCP flags (FIN or RST) -> stop indices
            for each stop index:
              4. drop flows without a defined start (SYN flag)
              5. calculate time gaps in flows -> time  
              6. combine properties of all flows: sum bytes, calculate duration - time, combine all TCP flags
              7. append aggregated flow 
          
          @param data_tcp: samples for unique tcp-5-tuple (np.array)
          '''
          # region -------------------------------------------- 1. filter flows based of first appearance on export host
          data_tcp = data_tcp[data_tcp[:, Feature.EXPORT_HOST] == data_tcp[0, Feature.EXPORT_HOST]] # select export host from first example for all examples
          # endregion
          # region ---------------------------------------------------------- 2. calculate time difference between flows
          time_diff = data_tcp[:, [Feature.FIRST_SWITCHED_TS, Feature.LAST_SWITCHED_TS]]   # get only first- and last-switched column
          time_diff = time_diff.reshape(-1, 1)                                             # reshape to one column
          time_diff = np.squeeze(time_diff)                                                # remove "empty" dimension
          time_diff = np.diff(time_diff)                                                   # calculate difference ls_n - fs_n, fs_n+1 - ls_n
          # endregion
          # region ---------------------------------------- 3. determine connection ends based on TCP flags (FIN or RST)
          tcp_flags = data_tcp[:, Feature.TCP_FLAGS]                                       # load only tcp_flags
          stop_indices = np.where(np.bitwise_and(tcp_flags, TCPFlags.FIN + TCPFlags.RST))  # find all stop indices with tcp flag  (FIN=1, RST=4)

          stop_indices = stop_indices[0]                                                   # get first sample with FIN or RST flag
          stop_indices = stop_indices.tolist()                                             # convert to index list
          start_index = 0                                                                  # init start index
          # endregion
          for stop_index in stop_indices:                                                  # for all samples from first
            # region -------------------------------------------------- 4. drop flows without a defined start (SYN flag)
            # discard tcp flows where first element do not start with SYN-flag
            if np.bitwise_and(data_tcp[start_index, Feature.TCP_FLAGS], TCPFlags.SYN) != TCPFlags.SYN: # SYN=2
              start_index = stop_index + 1 ####
              continue
            # endregion
            stop_index += 1            # increase stop index cause exclusive element selection
            # region ----------------------------------------------------------- 5. calculate time gaps in flows -> time
            time_diff_slice = time_diff[start_index * 2: (stop_index - 1) * 2]            # select all calculated time differences between samples
            i1 = np.where(time_diff_slice > 0)[0]                                         # select all time differences greater zero
            i2 = i1[i1 % 2 == 1]                                                          # select all time differences between samples
            time = np.sum(time_diff_slice[i2])                                            # sum all time differences
            if not time: time = 0.                                                        # if no time differences exists, set time to zero
            # endregion
            # region --- 6. combine properties of all flows: sum bytes, calculate duration - time, combine all TCP flags
            sum_bytes = np.sum(data_tcp[start_index: stop_index, Feature.BYTES])          # sum all bytes from start- to stop-index
            max_ls = np.max(data_tcp[start_index: stop_index, Feature.LAST_SWITCHED_TS])  # find max last switched time stamp
            tcp_flags = np.bitwise_or.reduce(data_tcp[start_index: stop_index, Feature.TCP_FLAGS], axis=0) # combine all tcp flags from all samples

            data_tcp[start_index, Feature.BYTES] = sum_bytes                              # update bytes for first sample
            data_tcp[start_index, Feature.TCP_FLAGS] = tcp_flags                          # update flags for first sample
            data_tcp[start_index, Feature.LAST_SWITCHED_TS] = max_ls                      # update last switched for first example
            duration = ((max_ls - data_tcp[start_index, Feature.FIRST_SWITCHED_TS]) - time) / 1000 # duration = last switched - first switched - time differences -> to seconds
            data_tcp[start_index, Feature.DURATION] = duration                            # update duration for first sample
            if duration > 0:
              data_tcp[start_index, Feature.BPS] = 8 * sum_bytes / duration
            # endregion
            # region ------------------------------------------------------------------------- 7. append aggregated flow
            aggregated_data.append(data_tcp[start_index])

            start_index = stop_index
            # endregion
        ##### ----------------------------

        # region ---------------------------------------------------------------- 6b. if flows belong to TCP (see tcp())
        if data[i, Feature.PROTOCOL] == 6: tcp(equal_5tuples)
        # endregion
        # region ---------------------------------------------------------------- 6c. if flows belong to UDP (see udp())
        elif data[i, Feature.PROTOCOL] == 17: udp(equal_5tuples)
        # endregion
        # region ------------------------------------------------------------------------------- 6d. neither TCP nor UDP
        else: aggregated_data += equal_5tuples.tolist()
        # endregion
      print('number of flows after aggregation(samples):', len(aggregated_data))
      return np.array(aggregated_data)

    @timing
    def aggregate_directions(data: np.array) -> np.array:
      ''' aggregate two flows based on direction (forward and reversed) currently not used
        1. select only 5-tuple 
        2. build mask to already visited elements
        for each flow: 
          3. set as visited
          4. invert flow (based on 5-tuple)
          5. search for reversed flows (only unvisited elements)
          6. for first inverse flow check timeout (AGGREGATION_TIMEOUT), else discard current flow
          7. collect possible labels (bytes, duration, bps) and append to current flow
          8. append aggregated flow
      
      @param data: data to aggregate (np.array)
      @return: aggregated directions (np.array)
      '''
      if not AGGREGATION_DIRECTIONS: return data

      # region ---------------------------------------------------------------------------------- 1. select only 5-tuple
      data_ = data[:, FIVE_TUPLE].astype(np.uint32) # get only 5-tuple
      # endregion
      # region --------------------------------------------------------------- 2. build mask to already visited elements
      mask_visited = np.ones((data.shape[0],), dtype=bool)

      aggregated_data = []
      # endregion
      for i, flow in enumerate(data_):
        # region ------------------------------------------------------------------------------------- 3. set as visited
        mask_visited[i] = False
        # endregion
        # region --------------------------------------------------------------------- 4. invert flow (based on 5-tuple)
        inverse_flow = np.array([
          flow[0], # protocol
          flow[2], # dst_addr
          flow[1], # src_addr
          flow[4], # dst_port
          flow[3], # src_port
          ])
        # endregion
        # region ------------------------------------------------ 5. search for reversed flows (only unvisited elements)
        mask = data_[:, ] == inverse_flow
        mask = np.all(mask, axis=1)

        mask_visited[np.argmin(mask)] = False

        inverse_flows = data[mask]
        # endregion
        # region -------------- 6. for first inverse flow check timeout (AGGREGATION_TIMEOUT), else discard current flow
        if len(inverse_flows) == 0: continue # discard flows without an inverse flow
        first_inverse_flow = inverse_flows[0]
        if first_inverse_flow[Feature.FIRST_SWITCHED_TS] - data[i, Feature.LAST_SWITCHED_TS] > AGGREGATION_TIMEOUT: continue
        # endregion
        # region -------------------------- 7. collect possible labels (bytes, duration, bps) and append to current flow
        values_from_inverse_flow = np.array([first_inverse_flow[Feature.BYTES],
                                             first_inverse_flow[Feature.DURATION],
                                             first_inverse_flow[Feature.BPS]])
        aggregated_flow = np.append(data[i], values_from_inverse_flow)
        # endregion
        # region ----------------------------------------------------------------------------- 8. append aggregated flow
        aggregated_data.append(aggregated_flow)
        # endregion

      print('number of flows after aggregation(directions):', len(aggregated_data))
      return np.array(aggregated_data)

    def remove_random_ports(data: np.array):
      ''' set random chosen port from the operating system (> 32767) to 0
      PORT_MAX_LIMIT define the threshold (if -1 port substitution is disabled) 
      
      @param data: input data (np.array)
      @return: data with substituted ports (np.array)
      '''
      if PORT_MAX_LIMIT == -1: return data
      src_mask = data[:, Feature.SRC_PORT] >= PORT_MAX_LIMIT
      data[src_mask, Feature.SRC_PORT] = 0

      dst_mask = data[:, Feature.DST_PORT] >= PORT_MAX_LIMIT
      data[dst_mask, Feature.DST_PORT] = 0
      return data

    num_block = 0
    num_flows_agg = 0

    for data in load_pickle('../datasets/' + DATASET):
      if len(data) == 0 : break

      if num_block % 50 > 0: # store each 50th block (for concept drift)
        num_block += 1
        continue

      data = format_data(data)
      data = filter_columns_and_remove_duplicates(data) # use default_filter_feature_list

      data = aggregate_samples(data)
      num_flows_agg += len(data)

      # data = aggregate_directions(data)
      data = remove_random_ports(data)

      data, labels = normalize_data(data)

      if False: # plot data distribution of labels and exit
        visualize_histogram(labels, 0, r'(a) number of bytes', bins=[0., 100., 200.], filename='bytes-{}'.format(num_block), color='#3366CC')
        visualize_histogram(labels, 1, r'(b) duration in $seconds$', bins=[0., 100., 200.], filename='duration-{}'.format(num_block), color='#CC3366')
        visualize_histogram(labels, 2, r'(c) bit rate in $^{bit}/_{sec}$', bins=[0., 500., 5000.], filename='bps-{}'.format(num_block), color='#66cc33')
        sys.exit()

      data_train, labels_train, data_test, label_test = split_train_test(data, labels, 10)
      data_train, labels_train = shuffle_data(data_train, labels_train)

      pickle_data(self.pickle_file, data_train, labels_train, data_test, label_test)
      # if num_block == 9: break # stop after the first 10 blocks
      num_block += 1
      print('num_flows_agg', num_flows_agg)

    print(num_block)
    print('num_flows_agg', num_flows_agg)
    sys.exit()

  def clean_up(self):
    ''' clean up function not used '''
    # nothing to cleanup
    pass


def visualize_histogram(labels, column, x_label, bins=None, filename='default', color='blue'):
  ''' plot labels as histogram (output: pdf) (50 bins) (log y-scale) 
  
  @param labels: input data of all labels (np.array)
  @param column: column which is selected from labels and plotted
  @param x_label: plot label of x-axis
  @param bins: (optional) list of boundaries for histogram (only conlsole output)
  @param filename: <filename>.pdf
  @param color: color for plot
  '''
  import matplotlib.pyplot as plt

  # activate LaTeX font
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')

  column = labels[:1000, column]

  if 'duration' in filename: # duration from milliseconds to seconds
    column = column // 1000

  median = np.median(column)
  min_ = np.min(column)
  max_ = np.max(column)
  mean = np.mean(column)

  print('=' * 100)
  print('column.shape', column.shape, type(column))
  print(x_label, '.min()', min_)
  print(x_label, '.max()', max_)
  print(x_label, '.median()', median)
  print(x_label, '.mean()', mean)
  if bins: print('hist column', np.histogram(column, bins=bins))
  print('=' * 100)

  _, ax = plt.subplots(figsize=(6, 4))                                  # create plot, ration 6:4
  plt.yscale('log')                                                     # log y-scale
  plt.ylabel(r'frequency', size=30)                                     # y axis label
  ax.hist(column, edgecolor='white', linewidth=1, bins=50, color=color) # plot histogram of data

  ax.annotate(# custom "legend" with median
    r'Median: {:3.0f}'.format(median),
    xy=(.95, 0.85),
    xycoords='axes fraction',
    fontsize=30,
    backgroundcolor='w',
    bbox=dict(
      boxstyle='round',
      fc='white',
      ec='black',
      alpha=0.8,
      ),
    **{'ha':'right'}
    )

  plt.xlim(0 - column.max() // 200, column.max() + column.max() // 200) # set limits for x axis
  plt.ylim(0.8, 10 ** 3)                                                # set limits for y axis

  ax.get_xaxis().set_major_locator(plt.MaxNLocator(4))                  # set 4 ticks

  # disable scientific notation e.g. 1e7 = 1000000
  ax.get_xaxis().get_major_formatter().set_useOffset(False)
  ax.get_xaxis().get_major_formatter().set_scientific(False)

  # set LaTeX half space as thousand separators
  ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: r'${:,}$'.format(int(x)).replace(',', '\,')))
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)

  ax.grid(True)
  plt.tight_layout(pad=1)
  plt.savefig('hist_{}.pdf'.format(filename))
  plt.clf()


if __name__ == '__main__':
  print('START')
  HS_Flows().get_data()
#     import numpy as np
#     data_train, labels_train, data_test, labels_test, properties = IPFix().get_data(clean_up=False)
#
#     print('data_train.shape', data_train.shape, 'labels_train.shape', labels_train.shape)
#     train = np.concatenate((data_train, np.expand_dims(labels_train, axis=1)), axis=1)
#     test = np.concatenate((data_test, np.expand_dims(labels_test, axis=1)), axis=1)
#
#     np.savetxt('train.csv', train, delimiter=';')
#     np.savetxt('test.csv', test, delimiter=';')
