import os
import tensorflow as tf

nnArchitecture = 57 #---------------------------------------

Nx = 600 # ------------- test
Ny = 5
Ns = 5 #--------------------- test
BatchSize = 64

CandleFile = "18-01-01-00-00-23-05-20-20-23-5m"
SmallSigma = 1
LargeSigma = 30
eFreeNoLog = True

nFiles_t = 10
nFiles_v = 10
n_readers = 10
shuffle_batch = 50  # Keep it small to speed up model loading. Sample_anchores already do static shuffling.
nPrefetch = tf.data.AUTOTUNE

dir_data = "/mnt/data/Trading/"
dir_datasets = os.path.join(dir_data, "Datasets", "model_{}".format(nnArchitecture))
dir_candles = os.path.join(dir_data, "Candles")

min_true_candle_percent_x = 40
chosen_markets_x = []
chosen_fields_x_names = ['ClosePrice'] #, 'BaseVolume']
min_true_candle_percent_y = 40
assert min_true_candle_percent_x == min_true_candle_percent_y
chosen_markets_y = []
chosen_fields_y_names = ['ClosePrice']

Standardization = True
Kill_Irregulars = True  # ----------------- pls implement it
Time_into_X = True
Time_into_Y = True
Transformer = True
Reuse_files = True
eFreeNoPlot = True

Epochs_Initial = 5000

Num_Layers = 8 # Wow
Num_Heads = 1   # As we have a single GPU, and we want to a exhaustic attention.
Factor_FF = 4
repComplexity = 8  # Wower
Dropout_Rate = 0.1

HuberThreshold = 1.0
CancleLossWeight = 1.0
TrendLossWeight = 50.0

dir_Checkpoint = os.path.join(dir_data, "Checkpoints")
checkpoint_filepath = os.path.join(dir_Checkpoint, "model_{}".format(nnArchitecture))
dir_CSVLogs = os.path.join(dir_data, "CSVLogs")
csvLogger_filepath = os.path.join(dir_CSVLogs, "model_{}".format(nnArchitecture))
Checkpoint_Monitor = "val_loss"
EarlyStopping_Min_Monitor = "val_loss"
EarlyStopping_Patience = 20

Learning_Rate = 0.001