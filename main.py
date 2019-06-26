from utils import *
from dataset.dataset import get_next_block
import defaultParser
import csv
import time

start = time.time()

print('TensorFlow version {}'.format(tf.__version__))
if tf.__version__ != '1.12.0': print('should be version 1.12', file=sys.stderr)

EVAL_STRUCTURES    = False # use t-SNE to evaluate input data (first 1000 data element of block 0 )
EVAL_LABELS        = False # analyze labels (distribution within classes for 10 blocks)
EVAL_CONCEPT_DRIFT = False # analyze concept drift (train and test on first block, only test on following)

# for test history
prev_data_test     = None
prev_labels_test   = None
                   
feature_denorm_fn  = None

# select features from dataset
feature_selector = [
  # 0,            # year
  1, 2, 3, 4, 5,  # month, day, hour, minute, second
  9, 10, 11,      # protocol, src_addr, dst_addr
  12, 13,         # src_network, dst_network
  14, 15,         # src_port, dst_port
  16, 17, 18, 19, # src_prefix_len, dst_pref_len src_asn, dst_asn
  20, 21, 22, 23, # src_longitude, dst_longitude, src_latitude, dst_latitude
  24, 25,         # src_country_code, dst_country_code
  26, 27,         # src_vlan, dst_vlan
  28, 29,         # src_locality, dst_locality
  30,             # tcp_flags
  ]

# data filter parameter
feature_filter = { x: None for x in range(31) }
feature_filter.update({
  # 0:  lambda x: bool(re.compile(r'').match(x)), # example only
  # 14: lambda x: x < 32768,                     # exclude ephemeral ports
  # 15: lambda x: x < 32768,                     # exclude ephemeral ports
  # 26: lambda x: x not in wlan_vlans,           # no WLAN
  # 27: lambda x: x not in wlan_vlans,           # no WLAN
  })


def class_labels(data, boundaries_BPS, boundaries_DURATION, properties):
  ''' convert class labels to one hot 
  
  @param data               : input data, all potential labels (np.array)
  @param boundaries_BPS     : class boundaries for bit rate (bps)
  @param boundaries_DURATION: class boundaries for duration (currently not used)
  @param properties         : dict, used to indicate (only aggregate_directions)

  @return classes in one hot format, number of classes
  '''
  class_dims       = list()
  class_collection = list()

  AGGREGATION_DIRECTIONS = properties.get('aggregation')[1]

  def create_label(data, bins):
    ''' classify the values into the individual classes
    
    @param data: input data, labels (np.array) 
    @param bins: list of boundaries
    '''
    class_dims.append(len(bins) - 1)
    data_    = data.astype(np.float32)
    classes_ = np.digitize(data_, bins) - 1
    class_collection.append(classes_)

  create_label(data[:, Labels.BPS], boundaries_BPS) # only use bit rate (bps) for class labeling
  ''' example: create 3 x 3 classes for labeling '''
  # create_label(data[:, Labels.DURATION], boundaries_DURATION)

  if AGGREGATION_DIRECTIONS: # currently not used
    create_label(data[:, Labels.REV_BPS], boundaries_BPS)
    # create_label(data[:, Labels.REV_DURATION], boundaries_DURATION)

  classes                             = np.zeros((data.shape[0], *class_dims))         # create class matrix with selected label dimensions
  index                               = np.arange(len(data))
  classes[index, (*class_collection)] = 1                                              # mark selected class
  _num_classes                        = np.prod(class_dims)
  classes                             = np.reshape(classes, (len(data), _num_classes)) # reshape to one hot
  elements_per_class                  = np.bincount(np.argmax(classes, axis=1))
  print('number of elements per class', elements_per_class)
  elements_per_class_all_blocks.append(elements_per_class)

  return classes, _num_classes


def split(data, labels, percent_test):
  ''' split the input data and labels into training data and labels and test data and labels by a given percentage

  @param data        : input data (np.array)
  @param labels      : input labels (np.array)
  @param percent_test: percent value (int) of the amount of test data/lables (default = 10%)

  @return: data_train (np.array), labels_train (np.array), data_test (np.array), labels_test (np.array)
  '''
  num_elements_train = int(data.shape[0] * ((100 - percent_test) / 100))

  data_test          = data[num_elements_train:]
  label_test         = labels[num_elements_train:]
                     
  data_train         = data[:num_elements_train]
  labels_train       = labels[:num_elements_train]

  return data_train, labels_train, data_test, label_test


def shuffle(data, labels):
  ''' shuffle data and labels 

  @param data  : input data (np.array)
  @param labels: input labels (np.array)

  @return: shuffled data (np.array), shuffled labels (np.array)
  '''
  permutation = np.random.RandomState(seed=42).permutation(data.shape[0])
  data        = data[permutation]
  labels      = labels[permutation]
  return data, labels


def prepare_block(save, FLAGS):
  ''' prepare a block of data 
  
      1. load train and test data, properties and parameter from dataset 
    1.1. (optional) collection class labels to calculate class distribution
      2. (optional) outlier detection for each individual block
      3. apply feature filters
      4. slice features via feature_selector
      5. (optional) create t-SNE plots
      6. (optional) create history of test data
      7. create tensorflow dataset object (tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet)
    
  @param save: dict from dataset (hs_flows) with: 'data_train', 'labels_train', 'data_test', 'labels_test', 
                                                  'properties', 'feature_list' and 'feature_denorm_fn'
  @param FLAGS: command line parameters from defaultParser
  @return: train (TF_DataSet), test (TF_DataSet), properties (dict)
  '''
  # region --------------------------------------------------------- 1. load data, properties and parameter from dataset
  boundaries_BPS      = FLAGS.boundaries_bps + [float('inf')]
  boundaries_DURATION = FLAGS.boundaries_duration + [float('inf')]
  history             = FLAGS.history
  cluster_centers     = FLAGS.cluster_centers

  properties          = save.get('properties')

  data_train          = save.get('data_train')
  labels_train        = save.get('labels_train')

  # region -------------------------------------------------------- 1.1 collection class labels to calculate distribution
  if EVAL_LABELS:
    print('add ', len(labels_train), 'values to class_labels')
    global class_labels_all_blocks
    class_labels_all_blocks.append(labels_train)
  # endregion

  labels_train, _num_classes = class_labels(labels_train, boundaries_BPS, boundaries_DURATION, properties)
  data_test                  = save.get('data_test')
  labels_test                = save.get('labels_test')
  labels_test, _,            = class_labels(labels_test, boundaries_BPS, boundaries_DURATION, properties)
  feature_list               = save.get('feature_list', None)

  global feature_denorm_fn # load feature_denorm_fn only from first block
  if not feature_denorm_fn: feature_denorm_fn = save.get('feature_denorm_fn', None)

  # region ---------------------------------------------------------------------------- OUTLIER DETECTION for each block
  if cluster_centers > 0:
    print('num samples before outlier detection', data_train.shape, data_test.shape)
    # combine training and test data for outlier detection
    data                         = np.vstack((data_train, data_test))
    labels                       = np.vstack((labels_train, labels_test))

    num_before_outlier_detection = len(data)

    # create kmeans cluster centers; return outlier indizes for filtering
    _, outlier_indizes           = kmeans(data, labels, FLAGS,              # data elements, labels only used for TF_DataSet
                                          first_n_elements=len(data_train), # all data in block
                                          outlier_detection=True,           # return cluster centers and outlier indizes
                                          )

    # build filter mask
    outlier_mask                  = np.ones((len(data),), dtype=bool)
    outlier_mask[outlier_indizes] = False
                                  
    # filter outliers             
    data                          = data[outlier_mask]
    labels                        = labels[outlier_mask]
    num_after_outlier_detection   = len(data)

    data_train, labels_train, data_test, labels_test = split(data, labels, 10)
    data, labels                                     = shuffle(data_train, labels_train)
    data, labels                                     = shuffle(data_test, labels_test)

    print('outlier reduction {} elements'.format(num_before_outlier_detection - num_after_outlier_detection))
  # endregion

  if FLAGS.class_weighting: # standard class weighting
    class_counter                       = np.bincount(np.argmax(labels_train, axis=1))
    inverse                             = 1. / class_counter
    max_                                = max(class_counter)
    train_class_weighting               = inverse * max_
    train_class_weighting              /= sum(train_class_weighting)
    properties['class_weighting_train'] = train_class_weighting

    class_counter                      = np.bincount(np.argmax(labels_test, axis=1))
    inverse                            = 1. / class_counter
    max_                               = max(class_counter)
    test_class_weighting               = inverse * max_
    test_class_weighting              /= sum(test_class_weighting)
    properties['class_weighting_test'] = test_class_weighting

  else: # undersampling
    mask      = np.zeros((labels_train.shape[0],), dtype=bool)
    min_count = min(np.bincount(np.argmax(labels_train, axis=1)))

    for current_class in range(3):
      class_indices               = np.argmax(labels_train, axis=1).astype(np.int)
      current_class_indices       = np.where(class_indices == current_class)[0][0:min_count]
      mask[current_class_indices] = True

    data_train                    = data_train[mask]
    labels_train                  = labels_train[mask]
                                  
    mask                          = np.zeros((labels_test.shape[0],), dtype=bool)
    min_count                     = min(np.bincount(np.argmax(labels_test, axis=1)))

    for current_class in range(3):
      class_indices               = np.argmax(labels_test, axis=1).astype(np.int)
      current_class_indices       = np.where(class_indices == current_class)[0][0:min_count]
      mask[current_class_indices] = True

    data_test   = data_test[mask]
    labels_test = labels_test[mask]

  print('class_distribution_labels_train: {}'.format(np.bincount(np.argmax(labels_train, axis=1))))
  print('class_distribution_labels_test: {}'.format(np.bincount(np.argmax(labels_test, axis=1))))
  properties['dimensions']  = (sum([ list(feature_list.values())[i] for i in feature_selector ]), 1)
  properties['num_classes'] = _num_classes

  print('num samples before', data_train.shape, data_test.shape)
  # endregion
  # region --------------------------------------------------------------------------------------------  print all lists
  print_feature_list(feature_list, feature_selector)
  print_feature_filter_function_list(feature_denorm_fn)
  # endregion
  # region ------------------------------------------------------------------------------------ 3. apply feature filters
  feature_filter_list = FLAGS.feature_filter
  for filter_expression in feature_filter_list: # '--feature_filters "1 ; lambda x: x < 10" "2 ; lambda y : y > 5"'
    if filter_expression == '': break
    exp         = eval(filter_expression.split(';')[1])
    feature_key = int(filter_expression.split(';')[0])
    feature_filter.update({feature_key: exp})

  print_feature_filter_list(feature_filter)

  feature_filter_index_list = [ (i, feature_key, feature_filter.get(str(i), None)) for i, feature_key in enumerate(feature_list) ]
  feature_filter_index_list = list(filter(lambda t: t[0] in feature_selector, feature_filter_index_list))
  feature_filter_index_list = list(filter(lambda t: t[2] != None, feature_filter_index_list))

  # feature filter based on defined expression
  for _, feature_name, lambda_exp in feature_filter_index_list:
    denorm_data_train = feature_denorm_fn[feature_name](data_train)
    mask              = np.squeeze(np.vectorize(lambda_exp)(denorm_data_train))
    data_train        = data_train[mask]
    labels_train      = labels_train[mask]

    denorm_data_test  = feature_denorm_fn[feature_name](data_test)
    mask              = np.squeeze(np.vectorize(lambda_exp)(denorm_data_test))
    data_test         = data_test[mask]
    labels_test       = labels_test[mask]
  # endregion
  # region ------------------------------------------------------------------------- slice features via feature_selector
  feature_indices  = list()
  current_position = 0
  for feature_index, feature_key in enumerate(feature_list): # user feature_list to create a list of indices
    feature_size = feature_list.get(feature_key)
    if feature_index in feature_selector:
      feature_indices += [ x for x in range(current_position, current_position + feature_size) ]
    current_position += feature_size

  print('feature_indices', feature_indices)
  # slice via feature_indices
  data_train = data_train[:, feature_indices]
  data_test  = data_test[:, feature_indices]
  # endregion
  # region --------------------------------------------------------------------------------------- 5. create t-SNE plots
  global block
  if EVAL_STRUCTURES and block == 0 and not EVAL_CONCEPT_DRIFT:
    tsne(data_test, labels_test, feature_denorm_fn, feature_list, FLAGS)
  # endregion
  # region ------------------------------------------------------------------------------ 6. create history of test data
  if history > 0:
    print('build test data with {}% history'.format(history))
    global prev_data_test, prev_labels_test

    # build test data with old test data from previous blocks (10%)
    if prev_data_test is not None:
      num_elements_test               = int(data_test.shape[0] * (history / 100))
      data_test[:num_elements_test]   = prev_data_test[:num_elements_test]
      labels_test[:num_elements_test] = prev_labels_test[:num_elements_test]
      data_test, labels_test          = shuffle(data_test, labels_test)

    prev_data_test   = data_test
    prev_labels_test = labels_test
  # endregion
  # region ------------------------------------------------------------------------- 7. create tensorflow dataset object
  print('num samples after', data_train.shape, data_test.shape)
  train = TF_DataSet(255. * data_train, labels_train, reshape=False)
  test  = TF_DataSet(255. * data_test, labels_test, reshape=False)
  # endregion

  return train, test, properties


def main(_):
  ''' build, train and test fc DNN 
  
    1. load/define variables for DNN (load first block)
      dnn build functions
    2. build DNN
  '''
  # region --------------------------------------------------------- 1. load/define variables for DNN (load first block)
  global feature_selector, block
  parser           = defaultParser.create_default_parser()
  FLAGS, _         = parser.parse_known_args()
  defaultParser.printFlags(FLAGS)
  
  output_filename  = FLAGS.output_file
  feature_selector = FLAGS.features
  
  iter_next        = get_next_block(FLAGS)
  save             = next(iter_next)
  
  train_dataset, test_dataset, properties = prepare_block(save, FLAGS) # load first block

  output_file     = open(output_filename, 'w')
  csv_writer      = csv.writer(output_file)

  LOG_FREQUENCY   = FLAGS.log_frequency
  BATCH_SIZE      = FLAGS.batch_size
  EPOCHS          = FLAGS.epochs
  LEARNING_RATE   = FLAGS.learning_rate
  DROPOUT_HIDDEN  = FLAGS.dropout_hidden
  DROPOUT_INPUT   = FLAGS.dropout_input
  LAYER_SIZES     = FLAGS.layers
  CLASS_WEIGHTING = FLAGS.class_weighting

  NUM_CLASSES     = properties['num_classes']
  INPUT_SIZE      = properties['dimensions'][0]

  # start an interactive session
  config                          = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.log_device_placement     = False
  sess                            = tf.InteractiveSession(config=config)
  
# placeholder for input variables
  x                = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
  y_               = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
  keep_prob_input  = tf.placeholder(tf.float32)
  keep_prob_hidden = tf.placeholder(tf.float32)

  class_weights    = tf.placeholder(tf.float32, shape=[NUM_CLASSES]) # only if class weighting is used
  # endregion

  def feed_dict(test=False, shuffle=False):
    ''' feed dict function including class weighting and class undersampling mechanism '''
    if not test:
      xs, ys = train_dataset.next_batch(BATCH_SIZE)
      k_h = DROPOUT_HIDDEN
      k_i = DROPOUT_INPUT
    else:
      xs, ys = test_dataset.next_batch(BATCH_SIZE, shuffle=shuffle)
      k_h = 1.0
      k_i = 1.0

    feed_dict_ = {
        x               : xs,
        y_              : ys,
        keep_prob_input : k_i,
        keep_prob_hidden: k_h,
        }

    if CLASS_WEIGHTING:
      if not test: feed_dict_[class_weights] = properties['class_weighting_train']
      else       : feed_dict_[class_weights] = properties['class_weighting_test']
    return feed_dict_

  # region ----------------------------------------------------------------------------------------- dnn build functions
  def weight_variable(shape, stddev):
    ''' create weight variable '''
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

  def bias_variable(shape):
    ''' create bias variable '''
    return tf.Variable(tf.zeros(shape))

  def fc_layer(x, channels_in, channels_out, stddev):
    ''' create a fully conntect layer (with ReLU) '''
    W   = weight_variable([channels_in, channels_out], stddev)
    b   = bias_variable([channels_out])
    act = tf.nn.relu(tf.matmul(x, W) + b)
    return act

  def logits_fn(x, channels_in, channels_out, stddev):
    ''' create logits '''
    W   = weight_variable([channels_in, channels_out], stddev)
    b   = bias_variable([channels_out])
    act = tf.matmul(x, W) + b
    return act
  # endregion

  # region ------------------------------------------------------------------------------------------------ 2. build DNN
  x_drop_inn = tf.nn.dropout(x, keep_prob_input)

  # input layer
  h_fc_prev  = fc_layer(x_drop_inn, INPUT_SIZE, LAYER_SIZES[0], stddev=1.0 / math.sqrt(float(INPUT_SIZE)))
  h_fc_prev  = tf.nn.dropout(h_fc_prev, keep_prob_hidden)

  for l, l_size in enumerate(LAYER_SIZES[1:]): # create hidden layers based on command line parameters
    h_fc_prev = fc_layer(h_fc_prev, LAYER_SIZES[l], l_size, 1.0 / math.sqrt(float(LAYER_SIZES[l])))
    h_fc_prev = tf.nn.dropout(h_fc_prev, keep_prob_hidden)

  # create output layer (a softmax linear classification layer_sizes for the outputs)
  logits             = logits_fn(h_fc_prev, LAYER_SIZES[-1], NUM_CLASSES, stddev=1.0 / math.sqrt(float(LAYER_SIZES[-1])))
  cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
  if CLASS_WEIGHTING:
    weight_factor = tf.gather(class_weights, tf.cast(tf.argmax(y_, 1), tf.int32))
    avg_loss      = tf.reduce_mean(cross_entropy_loss * weight_factor)
  else:              
    avg_loss      = tf.reduce_mean(cross_entropy_loss)
                     
  train_step            = tf.train.AdamOptimizer(LEARNING_RATE).minimize(avg_loss)
  correct_prediction_tr = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
  accuracy_tr           = tf.reduce_mean(tf.cast(correct_prediction_tr, tf.float32))
  # endregion

  tf.global_variables_initializer().run() # Initialize all global tensorflow variables

  num_tested = 0
  # region ---------------------------------------------------------------------------------------- training and testing
  while True: # terminated if last block is processed
    batches_per_epoch      = math.ceil(train_dataset.images.shape[0] / BATCH_SIZE) # how many batches per epoch
    batches_per_epoch_test = math.ceil(test_dataset.images.shape[0] / BATCH_SIZE)  # how many batches per epoch (test)

    for epoch in range(0, EPOCHS):
      predicted_labels = list()
      real_labels      = list()
      for batch in range(batches_per_epoch):
        # training
        if EVAL_CONCEPT_DRIFT: # for concept drift detection: only train on block 1
          if block < 1: sess.run(train_step, feed_dict=feed_dict())
        else:
          sess.run(train_step, feed_dict=feed_dict())

        # test
        if batch % LOG_FREQUENCY == 0:
          acc = 0.
          for _ in range(batches_per_epoch_test):
            # print(sess.run(correct_prediction_tr, feed_dict=feed_dict(test=True)))
            ac   = sess.run(accuracy_tr, feed_dict=feed_dict(test=True))
            acc += ac
          acc /= batches_per_epoch_test
          print('test accuracy at step: {} \t \t {}'.format(batch, acc))
          csv_writer.writerow([batch, acc])
          if EVAL_CONCEPT_DRIFT:
            if block > 0: break

        if EVAL_CONCEPT_DRIFT: continue

        if EVAL_STRUCTURES and block == 0 and epoch == EPOCHS - 1 and batch == batches_per_epoch - 1: # block 0 , last epoch, last batch
          # collect labels for "wrong labels" plot
          test_dataset._index_in_epoch = 0
          print('batches_per_epoch_test', batches_per_epoch_test)
          for _ in range(batches_per_epoch_test):
            fd          = feed_dict(test=True, shuffle=False)
            num_tested += BATCH_SIZE
            predicted_labels.append(sess.run(tf.argmax(logits, 1), feed_dict=fd))
            real_labels.append(sess.run(y_, feed_dict=fd).argmax(axis=1))

          # build confusion matrix
          confusion = np.zeros((NUM_CLASSES, NUM_CLASSES))
          for _ in range(batches_per_epoch_test):
            fd         = feed_dict(test=True)
            ce         = sess.run(tf.argmax(logits, 1), feed_dict=fd)
            y          = sess.run(y_, feed_dict=fd).argmax(axis=1)
            confusion += sess.run(tf.confusion_matrix(labels=y, predictions=ce, num_classes=NUM_CLASSES))

          plot_confusion_matrix(confusion,
                                [ str(x) for x in range(NUM_CLASSES) ],
                                FLAGS,
                                title='',
                                normalize=True,
                                pos=[block, epoch, batch],
                                )
      # for batch in range(batches_per_epoch)
    # for epoch in range(EPOCHS)

    if EVAL_STRUCTURES and len(real_labels) != 0 and len(predicted_labels) != 0:
      # combine collected real and predicted labels for all blocks
      real_labels      = np.concatenate(real_labels)
      predicted_labels = np.concatenate(predicted_labels)
      tsne(None, None, None, None, FLAGS, real_labels=real_labels, predicted_labels=predicted_labels)

    # load next block from dataset
    try:
      block                         += 1
      save                           = next(iter_next)
      train_dataset, test_dataset, _ = prepare_block(save, FLAGS)
      if EVAL_CONCEPT_DRIFT: EPOCHS  = 1 # for concept drift detection
    except StopIteration:
      break
  # endregion

  output_file.close()

  if EVAL_LABELS: evaluate_labels(FLAGS)
  print('used time: {}s'.format(time.time() - start))
  print('end:', time.time())


if __name__ == '__main__':
  print('start:', time.time())
  tf.app.run(main=main, argv=[sys.argv[0]])

