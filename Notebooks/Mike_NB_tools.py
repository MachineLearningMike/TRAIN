# Import 3rd-party frameworks.

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import time as tm
import math
from datetime import datetime, timedelta
from matplotlib import pyplot as plt


def ShowSingle(title, series):
    fig = plt.figure(figsize=(16,3))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    for line in series:
        ax.plot(line[0], label = line[1], color=line[2])
    ax.legend()
    plt.show()


def PoltNormalized(title, series, color = 'manual'):
    fig = plt.figure(figsize=(16,3))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    if len(series) > 1:
        min0 = np.min(series[0][0]); max0 = np.max(series[0][0])
        for i in range(len(series)):
            minV = np.min(series[i][0]); maxV = np.max(series[i][0])
            series[i][0] = (series[i][0]-minV) * (max0-min0) / (maxV-minV+1e-9)

    for line in series:
        if color == 'manual':
            ax.plot(line[0], label = line[1], color=line[2])
        else:
            ax.plot(line[0], label = line[1])

    ax.legend(loc = 'upper left')
    plt.show()


def Get_Event_Free_Data_Scheme_10(candles, smallSigma, largeSigma, nLatest):

    def gaussian( x, s): return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )

    smallSigma = min(math.floor(candles.shape[0]/3), smallSigma)
    smallP = 3 * smallSigma
    smallKernel = np.fromiter( (gaussian( x , smallSigma ) for x in range(-smallP+1, 1, 1 ) ), float ) # smallP points, incl 0.
#     print("smallKernel: {}".format(smallKernel))
    maP = np.convolve(candles[:,3], smallKernel, mode="valid") / np.sum(smallKernel) # maps to candles[smallP-1:]
    log_maP = np.log2(maP + 1e-9) # maps to candles[smallP-1:]

    largeSigma = min(math.floor(candles.shape[0]/3), largeSigma)
    largeP = 3 * largeSigma
    largeKernel = np.fromiter( (gaussian( x , largeSigma ) for x in range(-largeP+1, 1, 1 ) ), float ) # largeP points, incl 0.
#     print("largeKernel: {}".format(largeKernel))
    event = np.convolve(log_maP, largeKernel, mode="valid") / np.sum(largeKernel) # maps to log_maP[largeP-1:], so to candles[smallP+largeP-2:]

    assert event.shape[0] == candles.shape[0] - (smallP+largeP-2)
    log_maP1 = log_maP[largeP-1:] # maps to log_maP[largeP-1:], so to candles[smalP+largeP-2:]
    assert log_maP1.shape[0] == candles.shape[0] - (smallP+largeP-2)
    P1 = candles[smallP+largeP-2:, 3]
    assert P1.shape[0] == candles.shape[0] - (smallP+largeP-2)
    eventFree = log_maP1 - event # maps to candles[smallP+largeP-2:]

    nLatest = min(candles.shape[0] - (smallP+largeP-2), nLatest)
    P2 = P1[-nLatest:]
    maP2 = maP[-nLatest:]
    logP2 = np.log2(P2 + 1e-9)
    log_maP2 = log_maP1[-nLatest:]
    event2 = event[-nLatest:]
    eventFree2 = eventFree[-nLatest:] # maps to candle[p1-1+p2-1+begin: p1-1+p2-1+begine+width]

    return P2, maP2, logP2, log_maP2, event2, eventFree2    # eventFree = log_maP - event, event = convolve(lag_maP, leftKernel) / sum(leftKernel)


def Event_Free_Learning_Scheme_10(candles, smallSigma, largeSigma, nLatest):
    # Show a Gaussian-weighted left moving average of closing prices.
    P2, maP2, logP2, log_maP2, event2, eventFree2 = Get_Event_Free_Data_Scheme_10(candles, smallSigma, largeSigma, nLatest)

    minEF = np.min(eventFree2); maxEF = np.max(eventFree2)
    minP = np.min(P2); maxP = np.max(P2)
    P3 = (P2-minP) / max(maxP-minP, 1e-9) * (maxEF-minEF)
    minMP = np.min(maP2); maxMP = np.max(maP2)
    maP3 = (maP2-minMP) / max(maxMP-minMP, 1e-9) * (maxEF-minEF)
    minLP = np.min(logP2); maxLP = np.max(logP2)
    logP3 = (logP2-minLP) / max(maxLP-minLP, 1e-9) * (maxEF-minEF)
    minLMP = np.min(log_maP2); maxLMP = np.max(log_maP2)
    log_maP3 = (log_maP2-minLMP) / max(maxLMP-minLMP, 1e-9) * (maxEF-minEF)
    minE = np.min(event2); maxE = np.max(event2)
    event3 = (event2-minE) / max(maxE-minE, 1e-9) * (maxEF-minEF)
    eventFree3 = eventFree2 - minEF

    series = [
        [maP3, "maP", "g"], [logP3, "logP" ,"m"], [log_maP3, "log.maP", "b"], [event3, "event = MA(log.maP))", "c"], [eventFree3, "e.Free = log.maP - event", "brown"]
    ]
    PoltNormalized("Event-free (brown) series is relatively stable, vibrating around a fixed axis. Scaled to fit onto the chart.", series)


def Show_Price_Volume_10(candles, pSigma, vSigma, nLatest):

    """
    df[0] # Open
    df[1] # High
    df[2] # Low
    df[3] # Close
    df[4] # Volume
    df[5] # Quote asset volume
    df[6] # Number of trades
    df[7] # Taker buy base asset volume
    df[8] Taker buy quote asset volume
    df[9] # Ignore
    """

    def gaussian( x, s): return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )

    pSigma = min(math.floor(candles.shape[0]/3), pSigma)
    pP = 3 * pSigma
    pKernel = np.fromiter( (gaussian( x , pSigma ) for x in range(-pP+1, 1, 1 ) ), float ) # pP points, incl 0.
    # print("pKernel: {}".format(pKernel))
    maP = np.convolve(candles[:, 3], pKernel, mode="valid") / np.sum(pKernel) # maps to candles[smallP-1:]

    vSigma = min(math.floor(candles.shape[0]/3), vSigma)
    vP = 3 * vSigma
    vKernel = np.fromiter( (gaussian( x , vSigma ) for x in range(-vP+1, 1, 1 ) ), float ) # vP points, incl 0.
    # print("vKernel: {}".format(vKernel))
    maV = np.convolve(candles[:, 4], vKernel, mode="valid") / np.sum(vKernel) # maps to log_maP[vP-1:], so to candles[pP+vP-2:]
    maQV = np.convolve(candles[:, 5], vKernel, mode="valid") / np.sum(vKernel)
    maTBBV = np.convolve(candles[:, 7], vKernel, mode="valid") / np.sum(vKernel)
    maTBQV = np.convolve(candles[:, 8], vKernel, mode="valid") / np.sum(vKernel)

    maP2 = maP[-nLatest:]
    minP = np.min(maP2); maxP = np.max(maP2)
    maP3 = maP2 - minP

    maV2 = maV[-nLatest:]
    minV = np.min(maV2); maxV = np.max(maV2)
    maV3 = (maV2-minV) / max(maxV-minV, 1e-9) * (maxP-minP)

    maQV2 = maQV[-nLatest:]
    minQV = np.min(maQV2); maxQV = np.max(maQV2)
    maQV3 = (maQV2-minQV) / max(maxQV-minQV, 1e-9) * (maxP-minP)

    maTBBV2 = maTBBV[-nLatest:]
    minTBBV = np.min(maTBBV2); maxTBBV = np.max(maTBBV2)
    maTBBV3 = (maTBBV2-minTBBV) / max(maxTBBV-minTBBV, 1e-9) * (maxP-minP)

    maTBQV2 = maTBQV[-nLatest:]
    minTBQV = np.min(maTBQV2); maxTBQV = np.max(maTBQV2)
    maTBQV3 = (maTBQV2-minTBQV) / max(maxTBQV-minTBQV, 1e-9) * (maxP-minP)


    series = [  [maP3, "ma.Price", "r"], \
                [maV3, "ma.Volume", "brown"], \
                [maQV3, "ma.QuoteVolum", "cyan"], \
                [maTBBV3, "ma.TakerBuyBaseV", "green"], \
                [maTBQV3, "ma.TakerBuyQuoteV", "orange"] 
            ]
    ShowSingle("Price and Volume look independent. Scaled to fit onto the chart.", series)


#==================== Define 'intervalToMilliseconds' ====================

def intervalToMilliseconds(interval):
    ms = None
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms= int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms


#==================== Define 'Get_eFree' ====================

def Get_eFree(feature, smallSigma, largeSigma, nLatest):

    def gaussian( x, s): return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )

    smallSigma = min(math.floor(feature.shape[0]/3), smallSigma)
    smallP = 3 * smallSigma
    smallKernel = np.fromiter( (gaussian( x , smallSigma ) for x in range(-smallP+1, 1, 1 ) ), float ) # smallP points, incl 0.
#     print("smallKernel: {}".format(smallKernel))
    maP = np.convolve(feature, smallKernel, mode="valid") / np.sum(smallKernel) # maps to feature[smallP-1:]

    # maP = maP / np.min(1.0, np.min(maP[np.where(maP>0.0)]))
    nzPs = np.where(maP > 0.0)[0] [smallP:]    # to exclude initial nearly-zero values.
    log_maP = np.zeros( maP.shape, dtype=maP.dtype)
    log_maP[nzPs] = np.log2(maP[nzPs])  #------------------------------------------ Log danger ------------

    # log_maP = np.log2(maP + 1e-9) # maps to feature[smallP-1:]

    largeSigma = min(math.floor(feature.shape[0]/3), largeSigma)
    largeP = 3 * largeSigma
    largeKernel = np.fromiter( (gaussian( x , largeSigma ) for x in range(-largeP+1, 1, 1 ) ), float ) # largeP points, incl 0.
#     print("largeKernel: {}".format(largeKernel))
    event = np.convolve(log_maP, largeKernel, mode="valid") / np.sum(largeKernel) # maps to log_maP[largeP-1:], so to feature[smallP+largeP-2:]

    assert event.shape[0] == feature.shape[0] - (smallP+largeP-2)
    log_maP1 = log_maP[largeP-1:] # maps to log_maP[largeP-1:], so to feature[smalP+largeP-2:]
    assert log_maP1.shape[0] == feature.shape[0] - (smallP+largeP-2)
    P1 = feature[smallP+largeP-2:]
    assert P1.shape[0] == feature.shape[0] - (smallP+largeP-2)
    eventFree = log_maP1 - event # maps to feature[smallP+largeP-2:]

    nLatest = min(feature.shape[0] - (smallP+largeP-2), nLatest)
    P2 = P1[-nLatest:]
    maP2 = maP[-nLatest:]

    # P2 = P2 / np.min(1.0, np.min(P2[np.where(P2>0.0)]))
    nzPs = np.where(P2 > 0.0) [0] [:]
    logP2 = np.zeros( P2.shape, dtype=P2.dtype)
    logP2[nzPs] = np.log2(P2[nzPs]) #------------------------------------------ Log danger ------------

    # logP2 = np.log2(P2 + 1e-9) 
    
    log_maP2 = log_maP1[-nLatest:]
    event2 = event[-nLatest:]
    eventFree2 = eventFree[-nLatest:] # maps to candle[p1-1+p2-1+begin: p1-1+p2-1+begine+width]

    return P2, maP2, logP2, log_maP2, event2, eventFree2    # eventFree = log_maP - event, event = convolve(lag_maP, leftKernel) / sum(leftKernel)


def Get_eFree_noLog(feature, smallSigma, largeSigma, nLatest):

    def gaussian( x, s): return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )

    smallSigma = min(math.floor(feature.shape[0]/3), smallSigma)
    smallP = 3 * smallSigma
    smallKernel = np.fromiter( (gaussian( x , smallSigma ) for x in range(-smallP+1, 1, 1 ) ), float ) # smallP points, incl 0.
#     print("smallKernel: {}".format(smallKernel))
    maP = np.convolve(feature, smallKernel, mode="valid") / np.sum(smallKernel) # maps to feature[smallP-1:]

    largeSigma = min(math.floor(feature.shape[0]/3), largeSigma)
    largeP = 3 * largeSigma
    largeKernel = np.fromiter( (gaussian( x , largeSigma ) for x in range(-largeP+1, 1, 1 ) ), float ) # largeP points, incl 0.
#     print("largeKernel: {}".format(largeKernel))
    event = np.convolve(maP, largeKernel, mode="valid") / np.sum(largeKernel) # maps to maP[largeP-1:], so to feature[smallP+largeP-2:]

    assert event.shape[0] == feature.shape[0] - (smallP+largeP-2)
    P1 = feature[smallP+largeP-2:]
    assert P1.shape[0] == feature.shape[0] - (smallP+largeP-2)
    eventFree = maP[largeP-1:] - event # maps to feature[smallP+largeP-2:]

    nLatest = min(feature.shape[0] - (smallP+largeP-2), nLatest)
    P2 = P1[-nLatest:]
    maP2 = maP[-nLatest:]
   
    maP2 = maP[-nLatest:]
    event2 = event[-nLatest:]
    eventFree2 = eventFree[-nLatest:] # maps to candle[p1-1+p2-1+begin: p1-1+p2-1+begine+width]

    return P2, maP2, P2, maP2, event2, eventFree2    # eventFree = log_maP - event, event = convolve(lag_maP, leftKernel) / sum(leftKernel)


#==================== Define 'get_eFree_with_plot' ====================

def get_eFree_with_plot(market, field, feature, smallSigma, largeSigma, nLatest, noPlot = True, noLog = False):
    if noLog:
        P, maP, _, _, event, eventFree = Get_eFree_noLog(feature, smallSigma, largeSigma, nLatest)
        series = [ [maP, "maP", "g"], [event, "event", "c"],  [eventFree, "e.Free", "brown"] ] #, [P, "raw feature", "r"] ]
        if not noPlot:
            PoltNormalized("Event-free (brown) {} on {}".format(field, market), series)
        return P, maP, _, _, event, eventFree
    else:
        P, maP, logP, log_maP, event, eventFree = Get_eFree(feature, smallSigma, largeSigma, nLatest)
        series = [ [maP, "maP", "g"], [logP, "logP" ,"m"], [log_maP, "log.maP", "b"], [event, "event", "c"],  [eventFree, "e.Free", "brown"] ] #, [P, "raw feature", "r"] ]
        if not noPlot:
            PoltNormalized("Event-free (brown) {} on {}".format(field, market), series)
        return P, maP, logP, log_maP, event, eventFree


def get_timepoint_size(indices):
    size = 1
    for ids in indices:
        size *= len(ids)
    return size


import tensorflow as tf

#==================== Define 'divide_to_multiple_csv_files' ====================

def divide_to_multiple_csv_files(data, time_x, time_y, times, sample_anchors, name_prefix, nx, x_indices, ny, y_indices, header=None, n_parts=10):
    path_format = "{}_{:02d}.csv"

    filenames = []
    for file_idx, anchors in enumerate(np.array_split(sample_anchors, n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for anchor in anchors:
                x = np.reshape(data[anchor: anchor + nx][:, x_indices[0]][:, :, x_indices[1]], -1)
                if time_x is True:
                    assert times is not None
                    x_time = np.reshape(times[anchor: anchor + nx], -1)
                    x = np.concatenate((x, x_time))
                f.write(",".join([str(col) for col in x]))
                y = np.reshape(data[anchor + nx: anchor + nx + ny][:, y_indices[0]][:, :, y_indices[1]], -1)
                if time_y is True:
                    assert times is not None
                    y_time = np.reshape(times[anchor + nx: anchor + nx + ny], -1)
                    y = np.concatenate((y, y_time))
                f.write("," + ",".join([str(col) for col in y]))
                f.write("\n")
    return filenames

#==================== Define 'parse_csv_line_to_tensors' ====================

def parse_csv_line_to_tensors(line, nx, size_x, ny, size_y, time_x, time_y, size_time):
    span_x = nx * (size_x + (size_time if time_x else 0))
    span_y = ny * (size_y + (size_time if time_y else 0))
    defs = [tf.constant(0.0)] * (span_x + span_y)
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.reshape(fields[0: nx * size_x], [nx, -1])    # sequence of nx tokens, each of size_x
    if time_x:
        x_time = tf.reshape(fields[nx * size_x: span_x], [nx, -1])
        x = tf.concat([x, x_time], 1)
    y = tf.reshape(fields[span_x : span_x + ny * size_y], [-1])
    if time_y:
        y_time = tf.reshape(fields[span_x + ny * size_y : span_x + span_y], [-1])
        y = tf.concat([y, y_time], 0)
    return x, y

#==================== Define 'parse_csv_line_to_tensors' ====================

def parse_csv_line_to_tensors_for_transformer(line, nx, size_x, ny, size_y, time_x, time_y, size_time):
    span_x = nx * (size_x + (size_time if time_x else 0))
    span_y = ny * (size_y + (size_time if time_y else 0))
    defs = [tf.constant(0.0)] * (span_x + span_y)
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.reshape(fields[0: nx * size_x], [nx, -1])    # sequence of nx tokens, each of size_x
    if time_x:
        x_time = tf.reshape(fields[nx * size_x: span_x], [nx, -1])
        x = tf.concat([x, x_time], 1)
    y = tf.reshape(fields[span_x : span_x + ny * size_y], [ny, -1])
    if time_y:
        y_time = tf.reshape(fields[span_x + ny * size_y : span_x + span_y], [ny, -1])
        y = tf.concat([y, y_time], 1)

    x = tf.pad(x, [[1,1], [0,0]])   # Start, End
    y = tf.pad(y, [[1,1], [0,0]])   # Start, End

    if x.shape[-1] % 2 != 0:
        x = tf.pad(x, [[0,0], [0,1]])
        y = tf.pad(y, [[0,0], [0,1]])

    return (x, y[:-1]), y[1:]

#==================== Define 'csv_reader_to_dataset' ====================

def csv_reader_to_dataset(filenames, nx, size_x, ny, size_y, time_x, time_y, size_time, n_parse_threads=5, batch_size=32, shuffle_buffer_size=32*128, n_readers=5, transformer=False):
    dataset = tf.data.Dataset.list_files(filenames)
    # dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename), #.skip(1), as we have no headers.
        cycle_length=n_readers)
    dataset = dataset.shuffle(shuffle_buffer_size)          # Shuffle before batch
    if transformer:
        dataset = dataset.map(lambda x: parse_csv_line_to_tensors_for_transformer(x, nx, size_x, ny, size_y, time_x, time_y, size_time), num_parallel_calls=n_parse_threads)
    else:
        dataset = dataset.map(lambda x: parse_csv_line_to_tensors(x, nx, size_x, ny, size_y, time_x, time_y, size_time), num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size, drop_remainder=False)   # Batch the shuffled
    # dataset = dataset.shuffle(10)          # Shuffle again over batches.
    return dataset #.prefetch(3)

#========================================= MaskedHuber

def MaskedHuber_Core(y_true, y_pred, threshold=1.0):
    # y_true, y_pred: (batch, sequence, depth)
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error) / 2
    big_error_loss = threshold * (tf.abs(error) - threshold / 2)
    raw_loss = tf.where(is_small_error, small_error_loss, big_error_loss)
    mask = tf.cast(y_true != 0, dtype=y_pred.dtype)   # no need for 0.0
    masked_loss = tf.multiply(raw_loss, mask, name='masked_loss')
    rs_masked_loss = tf.reduce_sum(masked_loss, axis=-1, name='rs_masked_loss') # depth axis is reduced
    rs_mask = tf.reduce_sum(mask, axis=-1, name='rs_mask')  # depth axis is reduced
    loss = tf.divide(rs_masked_loss, rs_mask, name='loss')
    return loss


class MaskedHuber(tf.keras.losses.Loss):
    def __init__(self, name='mHuber', threshold=1.0):
        super(MaskedHuber, self).__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name)
        self.threshold = threshold
        
    def call(self, y_true, y_pred):
        loss = MaskedHuber_Core(y_true, y_pred)
        return loss


class MaskedHuber_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='mHuber', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_updates_seen = self.add_weight(name='num', initializer='zeros')
        self.avg_across_updates = self.add_weight(name='metric', initializer='zeros')

    def update_state(self, y_true, y_pred):
        loss = MaskedHuber_Core(y_true, y_pred)
        loss = tf.reduce_mean(loss, axis=None)
        self.num_updates_seen.assign_add(tf.constant(1, dtype=self.dtype))
        self.avg_across_updates.assign_add( (loss - self.avg_across_updates) / self.num_updates_seen )

    def result(self):
        return self.avg_across_updates
    
    def reset_states(self):
        self.num_updates_seen.assign(0.)
        self.avg_across_updates.assign(0.)

#========================================= MaskedMSE

def MaskedMSE_Core(y_true, y_pred):
        # y_true, y_pred: (batch, sequence, depth)
        raw_loss = tf.square(y_true - y_pred)
        mask = tf.cast(y_true != 0, dtype=y_pred.dtype, name='mask')   # no need for 0.0
        masked_loss = tf.multiply(raw_loss, mask, name='masked_loss')
        rs_masked_loss = tf.reduce_sum(masked_loss, axis=-1, name='rs_masked_loss')  # depth axis is reduced
        rs_mask = tf.reduce_sum(mask, axis=-1, name='rs_mask')  # depth axis is reduced
        loss = tf.divide(rs_masked_loss, rs_mask, name='loss')
        return loss


class MaskedMSE(tf.keras.losses.Loss):
    def __init__(self, name='mMSE'):
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name)
        
    def call(self, y_true, y_pred):
        return MaskedMSE_Core(y_true, y_pred)


class MaskedMSE_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='mMSE', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_updates_seen = self.add_weight(name='num', initializer='zeros')
        self.avg_across_updates = self.add_weight(name='metric', initializer='zeros')

    def update_state(self, y_true, y_pred):
        loss = MaskedMSE_Core(y_true, y_pred)
        loss = tf.reduce_mean(loss, axis=None)
        self.num_updates_seen.assign_add(tf.constant(1, dtype=self.dtype))
        self.avg_across_updates.assign_add( (loss - self.avg_across_updates) / self.num_updates_seen )

    def result(self):
        return self.avg_across_updates
    
    def reset_states(self):
        self.num_updates_seen.assign(0.)
        self.avg_across_updates.assign(0.)

#========================================= MaskedTrendError

def MaskedTrendError_Core(y_true, y_pred):
    # y_true, y_pred: (batch, sequence, depth)
    d_true = (y_true[:, 1:] - y_true[:, :-1])
    d_true = d_true / tf.expand_dims(tf.norm(d_true, ord='euclidean', axis=-1) + 1e-30, axis=-1)
    d_pred = (y_pred[:, 1:] - y_pred[:, :-1])
    d_pred = d_pred / tf.expand_dims(tf.norm(d_pred, ord='euclidean', axis=-1) + 1e-30, axis=-1)
    mask =  tf.cast(y_true != 0, dtype=y_pred.dtype)   # no need for 0.0
    mask = tf.multiply(mask[:, 1:], mask[:, :-1], name='mask')
    raw_loss = - d_true * d_pred    # What's better?
    raw_loss = tf.nn.relu(raw_loss)
    masked_loss = tf.multiply(raw_loss, mask, name='masked_loss')
    rs_masked_loss = tf.reduce_sum(masked_loss, axis=-1, name='rs_masked_loss')  # depth axis is reduced
    rs_mask = tf.reduce_sum(mask, axis=-1, name='rs_mask')   # depth axis is reduced
    loss = tf.divide(rs_masked_loss, rs_mask + 1e-30, name='loss')  # (batch, sequence-1)
    return loss


class MaskedTrendError(tf.keras.losses.Loss):
    def __init__(self, name='mTE'):
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name)
        
    def call(self, y_true, y_pred):
        loss = MaskedTrendError_Core(y_true, y_pred) # (batch, sequence-1)
        return loss


class MaskedTrendError_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='mTE', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_updates_seen = self.add_weight(name='num', initializer='zeros')
        self.avg_across_updates = self.add_weight(name='metric', initializer='zeros')

    def update_state(self, y_true, y_pred):
        loss = MaskedTrendError_Core(y_true, y_pred)
        loss = tf.reduce_mean(loss, axis=None)
        self.num_updates_seen.assign_add(1.)
        self.avg_across_updates.assign_add( (loss - self.avg_across_updates) / self.num_updates_seen )

    def result(self):
        return self.avg_across_updates
    
    def reset_states(self):
        self.num_updates_seen.assign(0.)
        self.avg_across_updates.assign(0.)


class MaskedTrendAccuracy_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='mTA', **kwargs):
        super().__init__(name=name, **kwargs)
        self.accMatch = self.add_weight(name='accMatch', initializer='zeros')
        self.accTotal = self.add_weight(name='accTotal', initializer='zeros')

    def update_state(self, y_true, y_pred):
        mask =  tf.cast(y_true != 0, dtype=y_pred.dtype)   # no need for 0.0
        mask = tf.multiply(mask[:, 1:], mask[:, :-1], name='mask')
        codir = (y_true[:, 1:] - y_true[:, :-1]) * (y_pred[:, 1:] - y_pred[:, :-1])
        masked_codir = tf.cast(tf.greater(codir, 0.0), self.dtype) * mask
        self.accMatch.assign_add(tf.reduce_sum(masked_codir, axis=None))
        self.accTotal.assign_add(tf.reduce_sum(mask, axis=None))

    def result(self):
        return self.accMatch / (self.accTotal + 1e-30)
    
    def reset_states(self):
        self.accMatch.assign(0.)
        self.accTotal.assign(0.)


#==================== Define plot_history ====================

def plot_history(history, loss="loss"):
    train_losses = history.history[loss]
    valid_losses = history.history["val_" + loss]
    n_epochs = len(history.epoch)
    minloss = min( np.min(valid_losses), np.min(train_losses) )
    maxloss = max( np.max(valid_losses), np.max(train_losses) )
    
    plt.title('History of loss')
    plt.plot(train_losses, color="b", label="Train")
    plt.plot(valid_losses, color="r", label="Validation")
    plt.plot([0, n_epochs], [minloss, minloss], "k--",
             label="Min val: {:.5f}".format(minloss))
    plt.axis([0, n_epochs, minloss/1.05, maxloss*1.05])
    plt.legend()
    plt.show()


def plot_log_history(history, loss="loss"):
    log_train_losses = np.log(history.history[loss])
    log_valid_losses = np.log(history.history["val_" + loss])
    n_epochs = len(history.epoch)
    minloss = min( np.min(log_valid_losses), np.min(log_train_losses) )
    maxloss = max( np.max(log_valid_losses), np.max(log_train_losses) )
    
    plt.title('History of logarithmic loss')
    plt.plot(log_train_losses, color="b", label="log.Train")
    plt.plot(log_valid_losses, color="r", label="log.Validation")
    plt.plot([0, n_epochs], [minloss, minloss], "k--",
             label="Min val: {:.5f}".format(minloss))
    plt.axis([0, n_epochs, minloss/1.05, maxloss*1.05])
    plt.legend()
    plt.show()


# # Find market clusters # temporary
# from sklearn.metrics import pairwise

# distances = np.zeros( (Candles.shape[1], Candles.shape[1]), dtype=float)

# # Find dependency distance
# for m in range(Candles.shape[1]):
#     distances[m, m] = 0.
#     for n in range(m+1, Candles.shape[1]):
#         mask = (CandleMarks[:, m] + CandleMarks[:, n] == 0) # CandleMarks == 0 : true full candles, CandleMarks = -1: price interpolated , CandleMarks = -2: coincodex prices
#         pm = Candles[mask, m, 0][np.newaxis]
#         pn = Candles[mask, n, 0][np.newaxis]
#         distances[m, n] = sklearn.metrics.pairwise.cosine_distances(pm, pn)
#         distances[n, m] = distances[m, n]

# from sklearn.cluster import OPTICS
# clustering = OPTICS(metric='precomputed', n_jobs=-1).fit(distances)
# print( clustering.labels_ )

# np.reshape(np.argwhere(clustering.labels_ == 1), -1)

# market_clusters = [ [ markets[ id ] for id in np.reshape(np.argwhere(clustering.labels_ == label), -1) ] for label in range(np.max(clustering.labels_))]
# print(market_clusters)

# cluster = 0
# ids = np.reshape(np.argwhere(clustering.labels_ == cluster), -1)
# series = [ [Candles[:, id, 0], markets[id] ] for id in ids ]
# PoltNormalized("Market cluster 0. Recent prices are mediocre. Shorter history.", series, color = 'auto')

# cluster = 1
# ids = np.reshape(np.argwhere(clustering.labels_ == cluster), -1)
# series = [ [Candles[:, id, 0], markets[id] ] for id in ids ]
# PoltNormalized("Market cluster 1. Vanished recently. Shorter history, Trends later", series, color = 'auto')

# cluster = 2
# ids = np.reshape(np.argwhere(clustering.labels_ == cluster), -1)
# series = [ [Candles[:, id, 0], markets[id] ] for id in ids ]
# PoltNormalized("Market cluster 2. Not vanished recently. Longer hostory. Trends earlier", series, color = 'auto')

