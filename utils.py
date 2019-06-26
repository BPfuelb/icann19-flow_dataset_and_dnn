'''
Created on 06.04.2019

load/install modules
helper functions (print functions, outlier detection function)
create plots and visualizations  
'''


def load_module(module, package=None):
  ''' auto module loader, if module not present, install into user space via pip  
  
  @param module: module name 
  @param package: (optional) package name to install if different from module name
  '''
  try:
    import pip
    if hasattr(pip, 'main'): from pip           import main as pip_install # linux
    else                   : from pip._internal import main as pip_install # windows
  except:
    raise Exception('install pip')
  try:
    import importlib
    importlib.import_module(module)
  except ImportError as ex:
    print(ex)
    if package is None: pip_install(['install', '--user', module])
    else              : pip_install(['install', '--user', package])
  finally:
    globals()[module] = importlib.import_module(module)


# required modules
modules = [
  'numpy'      ,
  'sys'        ,
  'math'       ,
  'itertools'  ,
  'abc'        ,
  'argparse'   ,
  'collections',
  'functools'  ,
  'importlib'  ,
  'os'         ,
  'scipy'      ,
  'shutil'     ,
  'time'       ,
  'gzip'       ,
  'pickle'     ,
  'zipfile'    ,
  'tarfile'    ,
  'pathlib'    ,
  'urllib'     ,
  'datetime'   ,
  'ipaddress'  ,
  'builtins'   ,
  'enum'       ,
  'csv'        ,
  ]

# auto install routine for modules
for module in modules: load_module(module)
load_module('safe_cast', 'safe-cast')
load_module('tensorflow', 'tensorflow-gpu==1.12') # gpu variant
# load_module('tensorflow', 'tensorflow==1.12')   # cpu variant: requires protobuf>=3.6.1

import numpy as np
import sys
import math
import itertools
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet as TF_DataSet

tsne_output = None
block       = 0

class Labels():
  ''' define positions (column) of labels in data '''
  BYTES        = 0
  DURATION     = 1
  BPS          = 2
  REV_BYTES    = 3
  REV_DURATION = 4
  REV_BPS      = 5


# structure to collect labels/element for evaluation
class_labels_all_blocks       = list()
elements_per_class_all_blocks = list()

np.set_printoptions(threshold=sys.maxsize) # print whole numpy arrays

# colors for t-SNE output
color_dict = {
  0:'red'    ,
  1:'blue'   ,
  2:'green'  ,
  3:'yellow' ,
  4:'purple' ,
  5:'grey'   ,
  6:'black'  ,
  7:'cyan'   ,
  8:'pink'   ,
  9:'skyblue',
  }
# markers for t-SNE output
marker_dict = {
  0:'o',
  1:'s',
  2:'P',
  3:'^',
  4:'X',
  5:'v',
  6:'*',
  7:'d',
  8:'D',
  9:'p',
  }

# region ---------------------------------------------------------------------------------- coloring parameter for t-SNE
wlan_vlans = np.array([3100, 3187, 3188, 3189, 3190, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199])
http_s     = np.array([80, 443])
dns        = np.array([53])
mail       = np.array([25, 465, 587, 993])
ftp        = np.array([20, 21])
sip_s      = np.array([5060, 5061])
snmp       = np.array([161])
dhcp       = np.array([67])
ldap       = np.array([636])
all_       = np.concatenate((http_s, dns)) # sip_s, mail, ftp, snmp, dhcp,ldap

feature_coloring = [
  # [ # Ports
  # ([14], lambda x: x < 1024, 'well-known port', 'blue', '^'),
  # ([14], lambda x: x >= 1024, 'dynamic port', 'red', '.'),
  # ],
  [  # transport-protocol
    ([9], lambda x     : x == 6, 'TCP', 'red', '^')             ,
    ([9], lambda x     : x == 17, 'UDP', 'cyan', 'o')           ,
    ([9], lambda x     : x not in [6, 17], 'other', 'blue', 's'),
  ],
  [  # intern/extern (private/public)
    ([28, 29], lambda x: x[0] == 0 and x[1] == 0, 'public/public', 'blue', '^')    ,
    ([28, 29], lambda x: x[0] == 0 and x[1] == 1, 'public/private', 'red', 'P')    ,
    ([28, 29], lambda x: x[0] == 1 and x[1] == 0, 'private/public', 'green', 'o')  ,
    ([28, 29], lambda x: x[0] == 1 and x[1] == 1, 'private/private', 'yellow', 's'),
  ],
  [  # wlan/non wlan
    ([26, 27], lambda x: np.bitwise_or(np.isin(x[0], wlan_vlans), np.isin(x[1], wlan_vlans)), 'WLAN', 'yellow', '*')                             , # WLAN
    ([26, 27], lambda x: np.bitwise_and(np.isin(x[0], wlan_vlans, invert=True), np.isin(x[1], wlan_vlans, invert=True)), 'no WLAN', 'green', 's'), # no WLAN
  ],
  [  # application protocol (based on src and dst port)
    ([14, 15], lambda x: np.bitwise_or(np.isin(x[0], http_s), np.isin(x[1], http_s)), 'HTTP(S)', 'yellow', 'd')                      ,
    ([14, 15], lambda x: np.bitwise_or(np.isin(x[0], dns), np.isin(x[1], dns)), 'DNS', 'red', 'o')                                   ,
    # disabled, because too many details
    # ([14, 15], lambda x: np.bitwise_or(np.isin(x[0], mail), np.isin(x[1], mail)), 'Mail', 'green', 'P'),
    # ([14, 15], lambda x: np.bitwise_or(np.isin(x[0], ftp), np.isin(x[1], ftp)), 'FTP', 'cyan', 's'),
    # ([14, 15], lambda x: np.bitwise_or(np.isin(x[0], sip_s), np.isin(x[1], sip_s)), 'SIP', 'blue', '+'),
    # ([14, 15], lambda x: np.bitwise_or(np.isin(x[0], snmp), np.isin(x[1], snmp)), 'SNMP', 'orange', 'h'),
    # ([14, 15], lambda x: np.bitwise_or(np.isin(x[0], dhcp), np.isin(x[1], dhcp)), 'DHCP', 'tomato', '<'),
    # ([14, 15], lambda x: np.bitwise_or(np.isin(x[0], ldap), np.isin(x[1], ldap)), 'LDAP', 'skyblue', '>'),
    ([14, 15], lambda x: np.bitwise_and(np.isin(x[0], all_, invert=True), np.isin(x[1], all_, invert=True)), 'other', 'skyblue', '^'),
  ],
  ]
# endregion


def print_feature_filter_list(feature_filter):
  ''' print feature filter list '''
  filter_list = [ (k, v) for k, v in feature_filter.items() if v != None ]
  if len(filter_list) == 0: return
  print('*' * 20, 'Feature Filter List', '*' * 20)
  for k, v in filter_list: print('key: {}  lambda expression: {}'.format(k, v))
  print('*' * 52)


def print_feature_list(feature_list, feature_selector):
  ''' print en- or disabled features '''
  print('*' * 20, 'Feature List', '*' * 20)
  for i, (k, v) in enumerate(feature_list.items()):
    print('{:<20} {:<4} #{:<2} {}'.format(k, v, i, '+' if i in feature_selector else ''))
  print('*' * 52)


def print_feature_filter_function_list(fn_list):
  ''' print feature filter function (partial functions) '''
  print('*' * 20, 'Feature Filter Function List', '*' * 20)
  for t in fn_list.items(): print(t)
  print('*' * 52)


def euclidian_distance(data, labels, c1, c2):
  ''' print euclidian distance of elements between different classes (currently not used)
    e.g.: 
      # euclidian_distance(xs, ys, 0, 1)
      # euclidian_distance(xs, ys, 1, 2)
      # euclidian_distance(xs, ys, 0, 2)
   
  @param data  : input data (np.array)
  @param labels: corresponding labels (np.array)
  @param c1    : first class (int)
  @param c2    : second class (int)
  '''
  print('first class', c1, 'second class + 1', c2)
  class_indices          = np.argmax(labels, axis=1).astype(np.int)
  current_class_indices0 = np.where(class_indices == c1)[0]
  data0                  = data[current_class_indices0]

  current_class_indices1 = np.where(class_indices == c2)[0]
  data1                  = data[current_class_indices1]

  print(c1, np.mean(np.sum((data0) ** 2, -1) ** 0.5))
  print(c2, np.mean(np.sum((data1) ** 2, -1) ** 0.5))
  print(c1, '-', c2, np.mean(np.sum((data0 - data1) ** 2, -1) ** 0.5))


def create_plot(filename, colors, marker, custom_lines, custom_labels, ncol=1, visual_clusters=None):
  ''' generic plot function for all t-SNE outputs
  
  @param filename       : <filename>_block_<>.pdf (str)
  @param colors         : a list of color strings for t-SNE output (list(str))
  @param marker         : list of marker strings for t-SNE output (list(str))
  @param custom_lines   : custom lines for legend (list(matplotlib.lines.Line2D))
  @param custom_lables  : custom labels for legend (list(str))
  @param ncol           : number of columns for legend (int)
  @param visual_clusters: if set, visualize cluster centers (list((avg_x, avg_y), cluster_points))
  '''
  import matplotlib.pyplot as plt
  global tsne_output
  fig = plt.figure(figsize=(12, 4)) # create new plot (ration 12:4)
  ax = fig.add_subplot(111)

  # set LaTeX font
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')

  for i, elem in enumerate(tsne_output): # draw each element in t-SNE output
    ax.scatter(
      elem[0]              , # x position of element
      elem[1]              , # y position of element
      c         = colors[i], # marker color
      marker    = marker[i], # marker shape (need loop because this)
      edgecolor = 'black'  , # marker color
      linewidth = '1'      , # marker edge width
      s         = 100      ,       # marker size
      )

  ax.legend(# add colored lines (markers) and text to legend
    custom_lines                , # use custom lines
    custom_labels               , # use custom labels
    prop          = {'size': 25}, # scale
    framealpha    = .9          , # alpha value
    ncol          = ncol        , # number of columns
    loc           = 2           , # position of legend (upper left)
    markerscale   = 1.2         , # marker scale
    borderaxespad = .2          , # vvv custom padding between elements
    columnspacing = .8          ,           
    handletextpad = .5          , # padding between marker and text
    handlelength  = .1          ,
    labelspacing  = .2          ,
    borderpad     = .4          ,
    )

  # disable tick labels
  ax.set_yticklabels([])
  ax.set_xticklabels([])

  # region ------------------------------------------- visualize cluster centers based on t-SNE positions (only k-means)
  if visual_clusters:
    for cluster_id in visual_clusters.keys(): # visual_cluster value structure [(avg_x, avg_y), cluster_points]
      avg_x  = visual_clusters[cluster_id][0][0]
      avg_y  = visual_clusters[cluster_id][0][1]
      points = visual_clusters[cluster_id][1]

      # draw cluster center
      plt.plot(avg_x, avg_y,
               color           = color_dict[cluster_id] ,
               marker          = marker_dict[cluster_id],
               linewidth       = 1                      ,
               markersize      = 30                     ,
               alpha           = 0.2                    ,
               markeredgecolor = 'black'                ,
               )
      # draw cluster-point connections
      for point in points:
        plt.plot([avg_x, point[0]]                 ,  # from x1 to x2
                 [avg_y, point[1]]                 ,  # from y1 to y2
                 linestyle = '-'                   ,
                 linewidth = 1                     ,
                 alpha     = 0.2                   ,
                 color     = color_dict[cluster_id],
                 )
  # endregion
  global block
  plt.tight_layout()
  plt.savefig('{}_block_{}.pdf'.format(filename, block))
  plt.clf() # reset plot for next plot


def kmeans(data, labels, FLAGS,
           first_n_elements=1000, num_clusters=None, outlier_detection=False, distance_metric_type=None):
  ''' calculate k-means for all elements at once TODO: use mini batch mode
  
  @param data                : input data (np.array)
  @param labels              : values only used as dummy (np.array)
  @param FLAGS               : use command line parameter from defaultParser (e.g., if set, use cluster_centers)
  @param first_n_elements    : number of selected elements (int) 
  @param outlier_detection   : if set, compute outlier detection (bool) (additional return value)
  @param num_clusters        : set the number of cluster centers (int)
  @param distance_metric_type: metric used for outlier detection (int)
    e.g: 0 = avg + std,
         1 = avg + (std/2)   
         2 = avg             
         3 = median + std    
         4 = median + (std/2)
         5 = median          
  @return the corresponding cluster id (np.array),(optional) outlier indices (np.array)
  '''
  BATCH_SIZE           = first_n_elements
  NUM_CLUSTERS         = FLAGS.cluster_centers if not num_clusters else num_clusters
  distance_metric_type = FLAGS.outlier_distance if not distance_metric_type else distance_metric_type
  EPOCHS               = 1  # no retraining

  data                 = data[:first_n_elements]
  labels               = labels[:first_n_elements] # TODO: use dummy data
  train_data           = TF_DataSet(data, labels, reshape=False)

  x                    = tf.placeholder(tf.float32, shape=[None, data.shape[1]])

  def feed_dict(): return {x: data}

  kmeans_tr = KMeans(
    inputs                        = x                  ,
    num_clusters                  = NUM_CLUSTERS       ,
    initial_clusters              = 'random'           , # "random", "kmeans_plus_plus", "kmc2", a tensor, or a function
    distance_metric               = 'squared_euclidean', # "squared_euclidean" or "cosine".
    use_mini_batch                = True               ,
    mini_batch_steps_per_iteration= BATCH_SIZE         , # batch size
    ) 

  all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, training_op = kmeans_tr.training_graph()
  '''
  0: all_scores,                  # A matrix (or list of matrices) of dimensions (num_input, num_clusters) where the value is the distance of an input vector and a cluster center. 
  1: cluster_idx,                 # A vector (or list of vectors). Each element in the vector corresponds to an input row in 'inp' and specifies the cluster id corresponding to the input. 
  2: scores,                      # Similar to cluster_idx but specifies the distance to the assigned cluster instead. 
  3: cluster_centers_initialized, # scalar indicating whether clusters have been initialized. 
  4: init_op,                     # an op to initialize the clusters. 
  5: training_op,                 # an op that runs an iteration of training.
  '''
  avg_distance = tf.reduce_mean(scores)

  init_vars    = tf.global_variables_initializer()
  sess         = tf.Session()
  sess.run(init_vars, feed_dict=feed_dict())

  initialized = False
  init_iteration = 0
  while(not initialized):
    sess.run(init_op, feed_dict=feed_dict())
    initialized = sess.run(cluster_centers_initialized)
    init_iteration += 1

  batches_per_epoch = math.ceil(train_data.images.shape[0] / BATCH_SIZE)  # how many batches per epoch
  for _ in range(EPOCHS):
    for _ in range(batches_per_epoch):
      sess.run(training_op, feed_dict=feed_dict())

  scores, _, idx, score = sess.run([all_scores, avg_distance, cluster_idx, scores], feed_dict=feed_dict())

  #--------------------------------------------------------------------------------------------------- OUTLIER DETECTION
  if outlier_detection:
    outlier_indices = []
    for cluster in range(NUM_CLUSTERS):
      cluster_distances = score[0][idx[0] == cluster]  # all distances for a cluster

      if   distance_metric_type == 0: cluster_dist = np.average(cluster_distances) + np.std(cluster_distances)       # avg + std
      elif distance_metric_type == 1: cluster_dist = np.average(cluster_distances) + (np.std(cluster_distances) / 2) # avg + (std/2)
      elif distance_metric_type == 2: cluster_dist = np.average(cluster_distances)                                   # avg
      elif distance_metric_type == 3: cluster_dist = np.median(cluster_distances) + np.std(cluster_distances)        # median + std
      elif distance_metric_type == 4: cluster_dist = np.median(cluster_distances) + (np.std(cluster_distances) / 2)  # median + (std/2)
      elif distance_metric_type == 5: cluster_dist = np.median(cluster_distances)                                    # median
      else:
        raise Exception('invalid distance_metric_type! ({})'.format(distance_metric_type))

      potential_outlier_indizes = np.nonzero(score[0] > cluster_dist)  # get all indices where distance is smaller than average
      cluster_indizes           = np.nonzero(idx[0] == cluster)        # get indices of cluster elements

      cluster_outlier           = np.intersect1d(cluster_indizes, potential_outlier_indizes) # get only indices which are outlier and in cluster (union)
      outlier_indices.append(cluster_outlier)                                                # append to outlier list

    outlier_indices             = np.concatenate(outlier_indices) # concatenate all outlier indices
    print('number outlier: {}'.format(len(outlier_indices)))
    return idx, outlier_indices

  return idx


def tsne(data, labels, feature_denorm_fn, feature_list, FLAGS,
         first_n_elements=1000, perplexity=50, learning_rate=200, n_iter=500,
         real_labels=None, predicted_labels=None):
  ''' calculate t-SNE (sklearn) and plot images (output is stored for a repetitive plot)
    1. (optional) wrong labels (done after training)
    2. class labels
    3. k-means output
    4. k-means with visualized centers and memberships
    5. feature coloring
    6. outlier dection (based on k-means)
    use create_plot function for all outputs
  
  @param data             : input data to plot (np.array)
  @param labels           : labels for class labels plot (np.array)
  @param feature_denorm_fn: denorm function to invert the normalization in 5. feature coloring 
  @param feature_list     : selected features for naming in 5. feature coloring
  @param FLAGS            : command line parameters (e.g., number of cluster centers)
  @param first_n_elements : slice first n element from data and labels 
  @param perplexity       : perplexity for t-SNE (int)
  @param learning_rate    : learn rate for t-SNE (int)
  @param n_iter           : number of iterations for t-SNE (int)
  @param real_labels      : (optional) real labels to plot 1. wrong labels (np.array) 
  @param predicted_labels : (optional) predicted labels to plot 1. wrong labels (np.array)
  '''
  if len(FLAGS.features) == 5: return # do not create plots for experiments with 5-tuple features

  from sklearn.manifold import TSNE
  from matplotlib.lines import Line2D

  global tsne_output, block

  def plot_wrong_labels(real_labels, predicted_labels):
    ''' plot wrong (and correct) classified samples '''
    real_labels      = real_labels[:first_n_elements]
    predicted_labels = predicted_labels[:first_n_elements]
    mask_correct     = real_labels == predicted_labels
    tsne             = tsne_output

    colors           = list()
    markers          = list()

    print('correct labels', np.sum(mask_correct))  # sum of correct labels
    wrong_labels_ = [0, 0, 0]                      # collect sum of false labels

    for i, label in enumerate(mask_correct):
      if label: # classification correct
        colors.append('white')
        markers.append('.')
      else: # classification wrong
        colors.append(color_dict[real_labels[i]])
        markers.append(marker_dict[real_labels[i]])
        wrong_labels_[real_labels[i]] += 1
    custom_lines  = list()
    custom_labels = list()

    print('wrong_labels_', wrong_labels_)

    # create legend markers (lines) and text
    for l in [('correct', '.', 'white'),
              ('c0 wrong', 'o', 'red'),
              ('c1 wrong', 's', 'blue'),
              ('c2 wrong', 'P', 'green'),
              ]:
      custom_lines.append(Line2D([], [], color=l[2], marker=l[1], markersize=15, lw=0, markeredgecolor='black'))
      custom_labels.append(l[0])

    create_plot('tsne_wrong_labels_features_all_n-elemts_{}_iter_{}'.format(first_n_elements, n_iter),
      colors, markers,
      custom_lines, custom_labels,
      ncol=2)
  #--------------------------------------------------------------------------------------------------- plot_wrong_labels

  if real_labels is not None and predicted_labels is not None: # is created after training process
    plot_wrong_labels(real_labels, predicted_labels)           # TODO: move all plots after training process?
    return

  # select the first n elements of a dataset
  data   = data[:first_n_elements]
  labels = labels[:first_n_elements]

  print('calculating TSNE for {} data elements'.format(data.shape[0]))

  global tsne_output # store output for coloring after training phase
  if not tsne_output:
    tsne = TSNE(
      n_components  = 2            ,
      perplexity    = perplexity   ,
      learning_rate = learning_rate,
      n_iter        = n_iter       ,
      ).fit_transform(data)
    tsne_output = tsne

  def plot_class_labels():
    ''' plot class labels (based on t-SNE) '''
    print('number of elements in each class', np.sum(labels, axis=0))
    num_classes   = np.argmax(labels, axis=1).max() + 1

    colors        = [ color_dict[label] for label in np.argmax(labels, axis=1) ]
    markers       = [ marker_dict[label] for label in np.argmax(labels, axis=1) ]

    custom_lines  = [ Line2D([], [], color=color_dict[c], marker=marker_dict[c], markersize=15, lw=0, markeredgecolor='black') for c in range(num_classes) ]
    custom_labels = [ 'c{}'.format(c) for c in range(num_classes) ]

    create_plot('tsne_class-labels_n-elemts_{}_iter_{}'.format(first_n_elements, n_iter),
      colors, markers,
      custom_lines, custom_labels,
      ncol=3,
      )
    #------------------------------------------------------------------------------------------------- plot_class_labels

  def plot_feature_coloring():
    ''' plot feature coloring (based on t-SNE) 
    
    e.g., mark Wi-fi samples 
    [...
      [ # wlan/non wlan
        ([26, 27], lambda x: np.bitwise_or(np.isin(x[0], wlan_vlans), np.isin(x[1], wlan_vlans)), 'WLAN', 'yellow', '*'),
        ([26, 27], lambda x: np.bitwise_and(np.isin(x[0], wlan_vlans, invert=True), np.isin(x[1], wlan_vlans, invert=True)), 'no WLAN', 'green', 's'),
      ],
    ]
      for each coloring list in feature coloring list 
        for each coloring element
          1. select features e.g., [26, 27] and apply the associated denorm function
          2. use lambda expression to build a filter mask for markers and colors
    '''
    for type_, coloring in enumerate(feature_coloring):
      custom_lines  = list()
      custom_labels = list()
      colors        = np.zeros((first_n_elements,), dtype=object)

      mask_other    = np.ones((data.shape[0],), dtype=bool)
      masks         = list()
      marker        = np.zeros((first_n_elements,), dtype=object)
      for feature_index, lambda_exp, name, color, marker_ in coloring:
        data_ = list()
        # region ---------------------------- 1. select features e.g., [26, 27] and apply the associated denorm function
        for index in feature_index:
          feature_name = list(feature_list)[index]
          data_.append(feature_denorm_fn[feature_name](data))
        # endregion
        denorm_data = np.concatenate((data_), axis=1)
        if len(denorm_data) != 0:
          # region ------------------------------ 2. use lambda expression to build a filter mask for markers and colors
          mask         = np.squeeze(np.apply_along_axis(lambda_exp, 1, denorm_data))
          print('elem: {} = {}'.format(name, np.sum(mask))) # only for evaluation
          mask_other   = np.bitwise_and(mask_other, np.invert(mask))
          marker[mask] = marker_
          masks.append((mask, name, color, marker_))

          custom_lines.append(Line2D([], [], color=color, marker=marker_, markersize=15, lw=0, markeredgecolor='black'))
          custom_labels.append(name)
          colors[mask] = color
          # endregion

        colors[mask_other] = 'black'
        marker[mask_other] = 's'

      create_plot('tsne_feature-coloring-{}_n-elemts_{}_iter_{}'.format(type_, first_n_elements, n_iter),
        colors, marker,
        custom_lines, custom_labels,
        ncol=[3, 2, 2, 3][type_])
    #--------------------------------------------------------------------------------------------- plot_feature_coloring

  def plot_kmeans(visualize_centers=False):
    ''' plot k-means output (based on t-SNE) 
    
    @param visualize_centers: if true, use t-SNE output (positions of clustered elements) to calculate a visual cluster center 
    '''
    num_clusters  = 10  # current only 10 colors and markers available (at more it becomes confusing)
    idx           = kmeans(data, labels, FLAGS, first_n_elements=first_n_elements, num_clusters=num_clusters)

    # translate cluster center ids into colors
    colors        = np.vectorize(lambda x: color_dict[x])(idx)[0]
    markers       = np.vectorize(lambda x: marker_dict[x])(idx)[0]

    # build legend content
    custom_lines  = [ Line2D([], [], color=color_dict[c], marker=marker_dict[c], markersize=15, lw=0, markeredgecolor='black') for c in range(num_clusters) ]
    custom_labels = [ 'cc{}'.format(c) for c in range(num_clusters) ]

    if visualize_centers: # build visual cluster centers for plot
      visual_clusters = dict()
      for cluster_id in range(num_clusters):                           # for each cluster center
        cluster_mask = [idx[0] == cluster_id]                          # select assigned elements (mask)
        all_cluster_points          = tsne_output[:, 0:2]              # select x and y position of elements
        cluster_points              = all_cluster_points[cluster_mask] # use mask to select corresponding elements (x,y)
        avg_x                       = np.mean(cluster_points[:, 0])    # calculate cluster center x position
        avg_y                       = np.mean(cluster_points[:, 1])    # calculate cluster center y position
        visual_clusters[cluster_id] = [(avg_x, avg_y), cluster_points] # build structure with position of cluster center and all corresponding points

      create_plot(
        'tsne_k-means-centers_n-elemts_{}_iter_{}'.format(first_n_elements, n_iter),
        colors, markers,
        custom_lines, custom_labels,
        ncol=2, visual_clusters=visual_clusters)
    else:
      create_plot(
        'tsne_k-means_n-elemts_{}_iter_{}'.format(first_n_elements, n_iter),
        colors, markers,
        custom_lines, custom_labels,
        ncol=2)

  def plot_kmeans_outlier():
    # calculate cluster center ids for original data and outlier
    for outlier_distance in range(1): # TODO: distance metric fixed to type 0
      # use k-means with 20 cluster centers for visualization
      _, outlier_indizes = kmeans(data, labels, FLAGS, first_n_elements=first_n_elements, num_clusters=20, outlier_detection=True, distance_metric_type=outlier_distance)

      colors                   = np.array(['#66cc33'] * len(data)) # normal
      markers                  = np.array(['o'] * len(data))
      colors[outlier_indizes]  = 'red'            # outlier
      markers[outlier_indizes] = 'X'

      # create custom legend marker/lines
      custom_lines = [Line2D([], [], color='#66cc33', marker='o', markersize=15, lw=0, markeredgecolor='black'),
                      Line2D([], [], color='red', marker='X', markersize=15, lw=0, markeredgecolor='black')    ,]
      custom_labels = ['normal', 'outlier']

      create_plot(
        'tsne_k-means_outlier_n-elemts_{}_iter_{}_centers_{}_distance_{}'.format(
        first_n_elements, n_iter, FLAGS.cluster_centers, outlier_distance),
        colors, markers,
        custom_lines, custom_labels,
        ncol=2,
        )

  plot_class_labels()
  plot_kmeans()
  plot_kmeans(visualize_centers=True)
  plot_feature_coloring()
  plot_kmeans_outlier()


def evaluate_labels(FLAGS):
  ''' plot class labels (only bit rate (BPS)) (collected over all processed blocks) 
    
    1. (re)use BPS boundaries for label division
    2. for each class:
      plot data distribution and calculate mean and median for each class
    3. calculate average of class elements (for all classes) 
   '''
  import matplotlib.pyplot as plt
  global class_labels_all_blocks

  # region ---------------------------------------------------------------- 1. (re)use BPS boundaries for label division
  data             = np.concatenate(class_labels_all_blocks)
  boundaries_BPS   = FLAGS.boundaries_bps + [float('inf')]

  class_dims       = list()
  class_collection = list()

  def create_label(data, bins):
    ''' classify the values into the individual classes (same function in main)
    
    @param data: input data, labels (np.array) 
    @param bins: list of boundaries
    '''
    class_dims.append(len(bins) - 1)
    data_    = data.astype(np.float32)
    classes_ = np.digitize(data_, bins) - 1
    class_collection.append(classes_)

  create_label(data[:, Labels.BPS], boundaries_BPS)
  # endregion
  classes = np.zeros((data.shape[0], *class_dims))  # create class matrix with selected label dimensions

  # region ----------------------------------------- 2. plot data distribution; calculate mean and median for each class
  for cls_ in range(3): # histogram for each class
    bps               = data[:, Labels.BPS]
    class_collection_ = np.concatenate(class_collection)
    cls_mask          = class_collection_ == cls_
    class_labels      = bps[cls_mask]
    fig               = plt.figure(figsize=(40, 10))
    ax                = fig.add_subplot(111)
    print('mean', np.mean(class_labels))    
    print('median', np.median(class_labels))
    plt.yscale('log')
    ax.hist(class_labels, edgecolor='white', linewidth=1, bins=50, color=color_dict[cls_])
    plt.xlim(boundaries_BPS[cls_], boundaries_BPS[cls_ + 1] if cls_ < 2 else np.max(class_labels))

    plt.axis('off')
    plt.savefig('label_distribution_cls_{}.pdf'.format(cls_), bbox_inches='tight')
    plt.clf()
  # endregion
  # region ---------------------------------------------------- 3. calculate average of class elements (for all classes)
  index                               = np.arange(len(data))
  classes[index, (*class_collection)] = 1                                              # mark selected class
  _num_classes                        = np.prod(class_dims)
  classes                             = np.reshape(classes, (len(data), _num_classes)) # reshape to one hot

  print('avg num elemes per class (10 blocks)', np.bincount(np.argmax(classes, axis=1)) / 10)

  global elements_per_class_all_blocks
  epcab = np.array(elements_per_class_all_blocks)[::2]
  print('elements_per_class_all_blocks', epcab)
  print('avg elements class weighting', np.average(epcab, axis=0))
  print('avg elements under sampling', np.average(np.min(epcab, axis=1), axis=0))
  # endregion


def plot_confusion_matrix(cm, classes, FLAGS, normalize=True, title='Confusion matrix', pos=[0, 0, 0]):
  ''' This function prints and plot the confusion matrix
  source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

  @param cm       : confusion matrix (np.array)
  @param classes  : name of the classes list(str)
  @param FLAGS    : command line arguments to distinguish plot colors
  @param normalize: if true, values are normalized (bool)
  @param title    : plot title (str)
  @param pos      : list of progress positions (list(block, epoch, batch))
  '''
  print('plot_confusion_matrix()')
  import matplotlib.pyplot as plt
  # reset open plot
  plt.cla()
  plt.clf()
  plt.close('all')

  # set LaTex font
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')

  print('Confusion matrix, without normalization')
  print(cm)

  if normalize: # normalize values
    print('Normalized confusion matrix')
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  else:
    print('Confusion matrix, without normalization')
  print(cm)

  # set color set (blue = all features, orange = 5-tuple)
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)  # @UndefinedVariable
  file_name = 'features_5-tuple'
  if len(FLAGS.features) > 5: # all features
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # @UndefinedVariable
    file_name = 'features_all'

  # set axis/plot texts
  plt.title(title)
  # plt.colorbar() # disable color legend
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, fontsize=20)
  plt.yticks(tick_marks, classes, fontsize=20)
  plt.ylabel('true label', size=24)
  plt.xlabel('predicted label', size=24)

  # positioning the values
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
      j, i, format(cm[i, j], '0.2f')                                 ,
      horizontalalignment = 'center'                                 ,
      verticalalignment   = 'center'                                 ,
      fontsize            = 34                                       ,
      color               = 'white' if cm[i, j] > thresh else 'black',
      )

  plt.tight_layout(pad=1)

  plt.savefig('{}_{}_norm_{}_block_{}_epoch_{}_batch_{}.pdf'.format('confustion_matrix', file_name, normalize, *pos))
  plt.clf()

