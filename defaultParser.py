
import argparse


def printFlags(flags):
    ''' print all flags '''
    side = '-' * int(((80 - len('Flags')) / 2))
    print(side + ' ' + 'Flags' + ' ' + side)
    for k, v, in sorted(vars(flags).items()):
      if v is not None:
        print('  * {:25}: {}'.format(k, v))
    print('-' * 80)


def create_default_parser():
    ''' create a parser with default parameters for most experiments '''
    parser = argparse.ArgumentParser()

    #------------------------------------------------------------------- DATASET
    parser.add_argument('--dataset_file'       , type=str  , default='HS_Flows.pkl.gz')
    parser.add_argument('--dataset_dir'        , type=str  , default='./datasets/')
    #-------------------------------------------------------- DROPTOUT PARAMETER
    parser.add_argument('--dropout_hidden'     , type=float, default=1)
    parser.add_argument('--dropout_input'      , type=float, default=1)
    #----------------------------------------------------------- LAYER PARAMETER
    parser.add_argument('--layers'             , type=int  , default=[800]             , nargs='+')
    #----------------------------------- BATCH_SIZE, LEARNING_RATE, EPOCHS, ETC.
    parser.add_argument('--batch_size'         , type=int  , default=100)
    parser.add_argument('--learning_rate'      , type=float, default=0.01)
    parser.add_argument('--epochs'             , type=int  , default=10)
    parser.add_argument('--log_frequency'      , type=int  , default=50)
    #------------------------------------------------------------------ FEATURES
    parser.add_argument('--features'           , type=int  , default=list(range(1, 31)), nargs='+', help='select features for training and testing') # 0 = year
    parser.add_argument('--class_weighting'    , type=bool , default=True                         , help='true=standard class weighting, false=under sampling')
    parser.add_argument('--feature_filter'     , type=str  , default=''                , nargs='*', help='set filter functions: "featurekey ; lambda x: <bool> " e.g.,"1 ; lambda x: x < 10" "2 ; lambda y : y > 5"')
    parser.add_argument('--boundaries_bps'     , type=float, default=[0., 500., 5000.] , nargs='+', help='set boundaries for bps')
    parser.add_argument('--boundaries_duration', type=float, default=[0., 100., 200.]  , nargs='+', help='set boundaries for duration')
    parser.add_argument('--history'            , type=float, default=0.                           , help='percent value to keep test data for next block, 0 = off')
    parser.add_argument('--cluster_centers'    , type=int  , default=0                            , help='define cluster center for outlier detection, 0 = off')
    parser.add_argument('--outlier_distance'   , type=int  , default=0, choices=range(6)          , help='0=avg + std,\n1=avg + (std/2),\n2=avg,\n3=median + std,\n4=median + (std/2),\n5=median,\n')
    #----------------------------------------------------- CSV OUTPUTS FOR PLOTS
    parser.add_argument('--output_file'        , type=str  , default='output.csv')

    return parser
