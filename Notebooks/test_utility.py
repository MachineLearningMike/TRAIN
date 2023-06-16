# Import 3rd-party frameworks.

import os
import math
import time as tm
import tensorflow as tf
from tensorflow import keras  # tf.keras
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

from test_trans import *

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
    smallKernel = np.fromiter( (gaussian( x , smallSigma ) for x in range(-smallP+1, 1, 1 ) ), candles.dtype ) # smallP points, incl 0.
#     print("smallKernel: {}".format(smallKernel))
    maP = np.convolve(candles[:,3], smallKernel, mode="valid") / np.sum(smallKernel) # maps to candles[smallP-1:]
    log_maP = np.log2(maP + 1e-9) # maps to candles[smallP-1:]

    largeSigma = min(math.floor(candles.shape[0]/3), largeSigma)
    largeP = 3 * largeSigma
    largeKernel = np.fromiter( (gaussian( x , largeSigma ) for x in range(-largeP+1, 1, 1 ) ), candles.dtype ) # largeP points, incl 0.
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
    pKernel = np.fromiter( (gaussian( x , pSigma ) for x in range(-pP+1, 1, 1 ) ), candles.dtype ) # pP points, incl 0.
    # print("pKernel: {}".format(pKernel))
    maP = np.convolve(candles[:, 3], pKernel, mode="valid") / np.sum(pKernel) # maps to candles[smallP-1:]

    vSigma = min(math.floor(candles.shape[0]/3), vSigma)
    vP = 3 * vSigma
    vKernel = np.fromiter( (gaussian( x , vSigma ) for x in range(-vP+1, 1, 1 ) ), candles.dtype ) # vP points, incl 0.
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


def get_timestamps(CandleFile, Candles):
    start = datetime( 2000+int(CandleFile[0:2]), int(CandleFile[3:5]), int(CandleFile[6:8]), int(CandleFile[9:11]), int(CandleFile[12:14]) )
    start_ts = round(datetime.timestamp(start))
    interval = CandleFile[ CandleFile.find('-', len(CandleFile) - 4) + 1 : ]
    interval_s = round(intervalToMilliseconds(interval) / 1000)
    timestamps_abs = np.array( range(start_ts, start_ts + Candles.shape[0] * interval_s, interval_s), dtype=np.int64) # must be 64.
    assert timestamps_abs.shape[0] == Candles.shape[0]
    return start_ts, interval_s, timestamps_abs


def get_timestamps_2(CandleFile, nCandles):
    start = datetime( 2000+int(CandleFile[0:2]), int(CandleFile[3:5]), int(CandleFile[6:8]), int(CandleFile[9:11]), int(CandleFile[12:14]) )
    start_ts = round(datetime.timestamp(start))
    interval = CandleFile[ CandleFile.find('-', len(CandleFile) - 4) + 1 : ]
    interval_s = round(intervalToMilliseconds(interval) / 1000)
    timestamps_abs = np.array( range(start_ts, start_ts + nCandles * interval_s, interval_s), dtype=np.int64) # must be 64
    assert timestamps_abs.shape[0] == nCandles
    return start_ts, interval_s, timestamps_abs

#==================== Define 'Get_eFree' ====================

def Get_eFree(feature, smallSigma, largeSigma, nLatest):

    def gaussian( x, s): return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )

    smallSigma = min(math.floor(feature.shape[0]/3), smallSigma)
    smallP = 3 * smallSigma
    smallKernel = np.fromiter( (gaussian( x , smallSigma ) for x in range(-smallP+1, 1, 1 ) ), feature.dtype ) # smallP points, incl 0.
#     print("smallKernel: {}".format(smallKernel))
    maP = np.convolve(feature, smallKernel, mode="valid") / np.sum(smallKernel) # maps to feature[smallP-1:]

    # maP = maP / np.min(1.0, np.min(maP[np.where(maP>0.0)]))
    nzPs = np.where(maP > 0.0)[0] [smallP:]    # to exclude initial nearly-zero values.
    log_maP = np.zeros( maP.shape, dtype=maP.dtype)
    log_maP[nzPs] = np.log2(maP[nzPs])  #------------------------------------------ Log danger ------------

    # log_maP = np.log2(maP + 1e-9) # maps to feature[smallP-1:]

    largeSigma = min(math.floor(feature.shape[0]/3), largeSigma)
    largeP = 3 * largeSigma
    largeKernel = np.fromiter( (gaussian( x , largeSigma ) for x in range(-largeP+1, 1, 1 ) ), feature.dtype ) # largeP points, incl 0.
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
    smallKernel = np.fromiter( (gaussian( x , smallSigma ) for x in range(-smallP+1, 1, 1 ) ), feature.dtype ) # smallP points, incl 0.
#     print("smallKernel: {}".format(smallKernel))
    maP = np.convolve(feature, smallKernel, mode="valid") / np.sum(smallKernel) # maps to feature[smallP-1:]

    largeSigma = min(math.floor(feature.shape[0]/3), largeSigma)
    largeP = 3 * largeSigma
    largeKernel = np.fromiter( (gaussian( x , largeSigma ) for x in range(-largeP+1, 1, 1 ) ), feature.dtype ) # largeP points, incl 0.
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


def get_formed_data( Candles, CandleMarks, all_market_names, all_field_names, 
        min_true_candle_percent_x, chosen_fields_names_x, min_true_candle_percent_y, chosen_fields_names_y,
        target_market_names, tarket_market_top_percent
    ):
    # marketrank, permute
    check = np.array([ np.argmax(Candles[:, m, 0]>0) / Candles.shape[0] * 100 for m in range(len(all_market_names)) ])
    permute = np.argsort(check)
    all_marketrank = [ (all_market_names[m], 100 - round(np.argmax(Candles[:, m, 0]>0) / Candles.shape[0] * 100)) for m in permute ]

    # permute all data.
    all_market_names = [all_market_names[m] for m in permute]
    Candles = Candles[:, permute]
    CandleMarks = CandleMarks[:, permute]

    # chosen_markets, chosen_fields
    chosen_market_names_x = [ elem[0] for elem in all_marketrank if elem[1] >= min_true_candle_percent_x ]
    chosen_markets_x = tuple([ all_market_names.index(elem) for elem in chosen_market_names_x ])
    chosen_markets_x = tuple(list(set(chosen_markets_x)))
    chosen_field_names_x = [ name for name in all_field_names if name in chosen_fields_names_x]
    chosen_fields_x = tuple( [ all_field_names.index(elem) for elem in chosen_field_names_x ] )
    chosen_fields_x = tuple(list(set(chosen_fields_x)))

    chosen_market_names_y = [ elem[0] for elem in all_marketrank if elem[1] >= min_true_candle_percent_y ]
    chosen_markets_y = tuple([ all_market_names.index(elem) for elem in chosen_market_names_y ])
    chosen_markets_y = tuple(list(set(chosen_markets_y)))
    chosen_field_names_y = [ name for name in all_field_names if name in chosen_fields_names_y]
    chosen_fields_y = tuple( [ all_field_names.index(elem) for elem in chosen_field_names_y ] )
    chosen_fields_y = tuple(list(set(chosen_fields_y)))

    chosen_markets = tuple(list(set(chosen_markets_x + chosen_markets_y)))
    chosen_market_names = [all_market_names[i] for i in chosen_markets]
    chosen_market_names = [ elem[0] for elem in all_marketrank if elem[0] in chosen_market_names ] # sort in rank
    chosen_fields = tuple(sorted(list(set(chosen_fields_x + chosen_fields_y))))  # sort in id
    chosen_field_names = [all_field_names[i] for i in chosen_fields]

    # Reduce Candles to chosen_markets and chosen_fields
    Candles = Candles[:][:, chosen_markets][:, :, chosen_fields]
    CandleMarks = CandleMarks[:][:, chosen_markets]

    # chosen_markets, chosen_fields. on chosen_market_names/chosen_field_names this time.
    # x_indices, y_indices
    chosen_market_names_x = [ elem[0] for elem in all_marketrank if elem[1] >= min_true_candle_percent_x ]
    chosen_markets_x = tuple([ chosen_market_names.index(elem) for elem in chosen_market_names_x ])
    chosen_markets_x = tuple(list(set(chosen_markets_x)))
    chosen_field_names_x = [ name for name in all_field_names if name in chosen_fields_names_x]
    chosen_fields_x = tuple( [ chosen_field_names.index(elem) for elem in chosen_field_names_x ] )
    chosen_fields_x = tuple(list(set(chosen_fields_x)))
    x_indices = ( chosen_markets_x, chosen_fields_x )

    chosen_market_names_y = [ elem[0] for elem in all_marketrank if elem[1] >= min_true_candle_percent_y ]
    chosen_markets_y = tuple([ chosen_market_names.index(elem) for elem in chosen_market_names_y ])
    chosen_markets_y = tuple(list(set(chosen_markets_y)))
    chosen_field_names_y = [ name for name in all_field_names if name in chosen_fields_names_y]
    chosen_fields_y = tuple( [ chosen_field_names.index(elem) for elem in chosen_field_names_y ] )
    chosen_fields_y = tuple(list(set(chosen_fields_y)))
    y_indices = ( chosen_markets_y, chosen_fields_y )

    # Compute target_markets/fields
    if target_market_names is not None:
        target_market_names = [m for m in target_market_names if m in chosen_market_names]
        target_markets = tuple([ chosen_market_names.index(elem) for elem in target_market_names ])
    elif tarket_market_top_percent is not None and tarket_market_top_percent > 0:
        target_market_names = chosen_market_names[: int(len(chosen_market_names) * tarket_market_top_percent / 100)]
        target_markets = tuple([ chosen_market_names.index(elem) for elem in target_market_names ])
    else:
        target_market_names = chosen_market_names_y
        target_markets = chosen_markets_y

    # Check for consistency
    marketrank = [ all_marketrank[i] for i in range(len(all_marketrank)) if all_marketrank[i][0] in chosen_market_names ]
    for r in marketrank:
        r[0] in chosen_market_names # obvious
        r[1] >= min(min_true_candle_percent_x, min_true_candle_percent_y)
    for r in all_marketrank:
        if r[1] >= min(min_true_candle_percent_x, min_true_candle_percent_y):
            assert r[0] in chosen_market_names

    return \
        Candles, CandleMarks, all_market_names, x_indices, y_indices, \
        chosen_market_names_x, chosen_field_names_x, chosen_market_names_y, chosen_field_names_y, \
        chosen_market_names, chosen_field_names, \
        target_market_names, target_markets


def get_formation_params(
        enFields, markets, marketrank,
        min_true_candle_percent_x, chosen_fields_x_names, min_true_candle_percent_y, chosen_fields_y_names
    ):
    chosen_markets_x = [ elem[0] for elem in marketrank if elem[1] >= min_true_candle_percent_x ]
    chosen_markets_x = tuple([ markets.index(elem) for elem in chosen_markets_x ])
    chosen_markets_x = tuple(list(set(chosen_markets_x)))

    chosen_fields_x = tuple( [ enFields.index(elem) for elem in chosen_fields_x_names ] )
    chosen_fields_x = tuple(list(set(chosen_fields_x)))
    x_indices = ( chosen_markets_x, chosen_fields_x )
    print(x_indices)

    chosen_markets_y = [ elem[0] for elem in marketrank if elem[1] >= min_true_candle_percent_y ]
    chosen_markets_y = tuple([ markets.index(elem) for elem in chosen_markets_y ])
    chosen_markets_y = tuple(list(set(chosen_markets_y)))

    chosen_fields_y = tuple( [ enFields.index(elem) for elem in chosen_fields_y_names ] )
    chosen_fields_y = tuple(list(set(chosen_fields_y)))
    y_indices = ( chosen_markets_y, chosen_fields_y )
    print(y_indices)

    size_x = get_timepoint_size(x_indices)
    size_y = get_timepoint_size(y_indices)
    print(size_x, size_y)

    chosen_markets = tuple(list(set(chosen_markets_x + chosen_markets_y)))
    chosen_fields = tuple(list(set(chosen_fields_x + chosen_fields_y)))
    print(chosen_markets, chosen_fields)

    print(len(chosen_markets), len(chosen_fields))

    return x_indices, y_indices, chosen_markets, chosen_fields


def get_formation_params_2(
        enFields, markets, marketrank,
        min_true_candle_percent_x, chosen_fields_x_names, min_true_candle_percent_y, chosen_fields_y_names
    ):
    chosen_markets_names_x = [ elem[0] for elem in marketrank if elem[1] >= min_true_candle_percent_x ]
    chosen_markets_x = tuple([ markets.index(elem) for elem in chosen_markets_names_x ])
    chosen_markets_x = tuple(list(set(chosen_markets_x)))

    chosen_fields_x = tuple( [ enFields.index(elem) for elem in chosen_fields_x_names ] )
    chosen_fields_x = tuple(list(set(chosen_fields_x)))

    chosen_markets_names_y = [ elem[0] for elem in marketrank if elem[1] >= min_true_candle_percent_y ]
    chosen_markets_y = tuple([ markets.index(elem) for elem in chosen_markets_names_y ])
    chosen_markets_y = tuple(list(set(chosen_markets_y)))

    chosen_fields_y = tuple( [ enFields.index(elem) for elem in chosen_fields_y_names ] )
    chosen_fields_y = tuple(list(set(chosen_fields_y)))
    y_indices = ( chosen_markets_y, chosen_fields_y )

    chosen_markets = tuple(list(set(chosen_markets_x + chosen_markets_y)))
    chosen_fields = tuple(list(set(chosen_fields_x + chosen_fields_y)))

    return chosen_markets_names_x, chosen_markets_names_y, chosen_markets, chosen_fields


def get_formation_params_3(
        enFields, markets, 
        chosen_markets_names_x, chosen_fields_x_names, 
        chosen_markets_names_y, chosen_fields_y_names
    ):
    chosen_markets_x = tuple([ markets.index(elem) for elem in chosen_markets_names_x ])
    chosen_markets_x = tuple(list(set(chosen_markets_x)))

    chosen_fields_x = tuple( [ enFields.index(elem) for elem in chosen_fields_x_names ] )
    chosen_fields_x = tuple(list(set(chosen_fields_x)))
    x_indices = ( chosen_markets_x, chosen_fields_x )

    chosen_markets_y = tuple([ markets.index(elem) for elem in chosen_markets_names_y ])
    chosen_markets_y = tuple(list(set(chosen_markets_y)))

    chosen_fields_y = tuple( [ enFields.index(elem) for elem in chosen_fields_y_names ] )
    chosen_fields_y = tuple(list(set(chosen_fields_y)))
    y_indices = ( chosen_markets_y, chosen_fields_y )

    return x_indices, y_indices

def get_time_features(timestamps_abs):
    sigma = np.power(2.0, -0.2)
    hourly = np.sin( 2 * np.pi / (60*60) * timestamps_abs ) / sigma
    daily = np.sin( 2 * np.pi / (60*60*24) * timestamps_abs ) / sigma
    weekly = np.sin( 2 * np.pi / (60*60*24*7) * timestamps_abs ) / sigma
    yearly = np.sin( 2 * np.pi / (60*60*24*365) * timestamps_abs ) / sigma
    tenyearly = np.sin( 2 * np.pi / (60*60*24*365*10) * timestamps_abs ) / sigma    # Let the model absorb non-cyclic time features.
    Time = np.stack([hourly, daily, weekly, yearly], axis=1)
    return Time


def standardize(Data, chosen_markets, chosen_fields):
    Standard = []

    for market in chosen_markets:
        _Standard = []
        for field in chosen_fields:
            nzPs = np.where( Data[:, market, field] != 0.0 ) [0]
            mu = np.average(Data[nzPs, market, field])
            sigma = np.std(Data[nzPs, market, field])
            standard = (Data[nzPs, market, field] - mu) / (sigma + 1e-15)
            _Standard.append( (market, field, mu, sigma) )
            Data[nzPs, market, field] = standard
        Standard.append(_Standard)
    Standard = np.array(Standard)
    return Data, Standard


def standardize_2(Candles):
    Standard = []

    for market in range(Candles.shape[1]):
        subStandard = []
        for field in range(Candles.shape[2]):
            nzPs = np.where( Candles[:, market, field] != 0.0 ) [0]
            mu = np.average(Candles[nzPs, market, field])
            sigma = np.std(Candles[nzPs, market, field])
            standardized = (Candles[nzPs, market, field] - mu) / (sigma + 1e-15)
            subStandard.append( (market, field, mu, sigma) )
            assert standardized.dtype == Candles.dtype
            Candles[nzPs, market, field] = standardized
        Standard.append(subStandard)
    Standard = np.array(Standard)
    return Candles, Standard


def get_sample_anchores(Data, Nx, Ny, Ns):
    sample_anchors = np.array(range(0, Data.shape[0] - Nx - Ny + 1, Ns))
    # print(Data.shape[0], len(sample_anchors), sample_anchors, sample_anchors[-1])
    # print(Data.shape[0], sample_anchors[ -1 ], sample_anchors[ -1 ] + Nx + Ny, sample_anchors[ -1 ] + Ns, sample_anchors[ -1 ] + Ns + Nx + Ny)

    for _ in range(100):
        permute = np.random.permutation(sample_anchors.shape[0])
        sample_anchors = sample_anchors[permute]

    sample_anchores_t, sample_anchores_v = train_test_split(sample_anchors, test_size=0.30, random_state=42)

    sample_anchores_t = tuple(sample_anchores_t)
    sample_anchores_v = tuple(sample_anchores_v)

    return sample_anchores_t, sample_anchores_v

def get_sample_anchors_2(Data, Nx, Ny, Ns):
    sample_anchors = np.array(range(0, Data.shape[0] - Nx - Ny + 1, Ns), dtype=np.int32)
    # print(Data.shape[0], len(sample_anchors), sample_anchors, sample_anchors[-1])
    # print(Data.shape[0], sample_anchors[ -1 ], sample_anchors[ -1 ] + Nx + Ny, sample_anchors[ -1 ] + Ns, sample_anchors[ -1 ] + Ns + Nx + Ny)

    for _ in range(100):
        permute = np.random.permutation(sample_anchors.shape[0])
        sample_anchors = sample_anchors[permute]

    sample_anchores_t, sample_anchores_v = train_test_split(sample_anchors, test_size=0.30, random_state=42)

    return sample_anchores_t, sample_anchores_v

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
                if time_x is True:
                    assert times is not None
                    y_time = np.reshape(times[anchor + nx: anchor + nx + ny], -1)
                    if time_y is True:
                        pass
                    else:
                        y_time[:] = 0.0    # This is a placeholder for no-info. Losses and metrics will recognize it.
                    y = np.concatenate((y, y_time))
                f.write("," + ",".join([str(col) for col in y]))
                f.write("\n")
    return filenames

#==================== Define 'parse_csv_line_to_tensors' ====================

def parse_csv_line_to_tensors(line, nx, size_x, ny, size_y, time_x, time_y, size_time):
    span_x = nx * (size_x + (size_time if time_x else 0))
    span_y = ny * (size_y + (size_time if time_x else 0))   # not time_y
    defs = [tf.constant(0.0)] * (span_x + span_y)
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.reshape(fields[0: nx * size_x], [nx, -1])    # sequence of nx tokens, each of size_x
    if time_x:
        x_time = tf.reshape(fields[nx * size_x: span_x], [nx, -1])
        x = tf.concat([x, x_time], 1)
    y = tf.reshape(fields[span_x : span_x + ny * size_y], [-1])
    if time_x:  # not time_y
        y_time = tf.reshape(fields[span_x + ny * size_y : span_x + span_y], [-1])
        y = tf.concat([y, y_time], 0)
    return x, y

#==================== Define 'parse_csv_line_to_tensors' ====================

def parse_csv_line_to_tensors_for_transformer(line, nx, size_x, ny, size_y, time_x, time_y, size_time):
    span_x = nx * (size_x + (size_time if time_x else 0))
    span_y = ny * (size_y + (size_time if time_x else 0))   # not time_y
    defs = [tf.constant(0.0)] * (span_x + span_y)
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.reshape(fields[0: nx * size_x], [nx, -1])    # sequence of nx tokens, each of size_x
    if time_x:
        x_time = tf.reshape(fields[nx * size_x: span_x], [nx, -1])
        x = tf.concat([x, x_time], 1)
    y = tf.reshape(fields[span_x : span_x + ny * size_y], [ny, -1])
    if time_x:  # not time_y
        y_time = tf.reshape(fields[span_x + ny * size_y : span_x + span_y], [ny, -1])
        y = tf.concat([y, y_time], 1)

    x = tf.pad(x, [[1,1], [0,0]], constant_values=0)   # (1 pre-pad: Start, 1 post-pad: End) on axis 0. (0 pre-pad, 0 post-pad) on axis 1.
    y = tf.pad(y, [[1,1], [0,0]], constant_values=0)

    if x.shape[-1] % 2 != 0:
        x = tf.pad(x, [[0,0], [0,1]], constant_values=0) # (0 pre-pad: Start, 0 post-pad: End) on axis 0. (0 pre-pad, 1 post-pad) on axis 1.
        y = tf.pad(y, [[0,0], [0,1]], constant_values=0)

    return (x, y[:-1]), y[1:]
    # so M(x, [y[0]]) -> y[1], M(x, [y[0], y[1]]) -> y[2], ..., M(x, [y[0], ..., y[-2]]) -> y[-1]
    # where y[0] = Start, y[-1] = End.

#==================== Define 'csv_reader_to_dataset' ====================

def csv_reader_to_dataset(filenames, nx, size_x, ny, size_y, time_x, time_y, size_time, n_parse_threads=5, batch_size=32, shuffle_buffer_size=32*128, n_readers=5, transformer=False):
    dataset = tf.data.Dataset.list_files(filenames)
    # dataset = dataset.repeat()
    dataset = dataset \
        .interleave(
            lambda filename: tf.data.TextLineDataset(filename), #.skip(1), as we have no headers.
            num_parallel_calls=tf.data.AUTOTUNE,    # num_parallel_calls
            cycle_length=n_readers
        ) \
        .prefetch(tf.data.AUTOTUNE) \
        .cache() \
        .shuffle(shuffle_buffer_size) \

    mapFun = parse_csv_line_to_tensors_for_transformer if transformer else parse_csv_line_to_tensors

    # shuffle before batch, batch before map

    dataset = dataset \
        .map(
            lambda x: mapFun(x, nx, size_x, ny, size_y, time_x, time_y, size_time),
            num_parallel_calls=tf.data.AUTOTUNE,    # num_parallel_calls
        ) \
        .batch(batch_size, drop_remainder=False) \
        .prefetch(tf.data.AUTOTUNE) \
        .cache()

    return dataset


def get_datasets(
    Reuse_files,
    CandleFile, dir_datasets, Data, Time_into_X, Time_into_Y, Time, 
    sample_anchores_t, sample_anchores_v,
    Nx, x_indices, Ny, y_indices, nFiles_t, nFiles_v, n_readers, size_time,
    BatchSize, shuffle_batch, Transformer, nPrefetch
       
):
    name_plus_t = CandleFile+'_t'
    name_plus_v = CandleFile+'_v'
    name_prefix_t = os.path.join(dir_datasets, name_plus_t)
    name_prefix_v = os.path.join(dir_datasets, name_plus_v)

    reuse_files = Reuse_files #------------------------------------------------------------------------------------------------------- 

    if reuse_files:
        import re
        filenames_train = [ os.path.join(dir_datasets, x) for x in os.listdir(dir_datasets) if re.match(name_plus_t, x)]
        filenames_valid = [ os.path.join(dir_datasets, x) for x in os.listdir(dir_datasets) if re.match(name_plus_v, x)]
    else:
        os.system("rm {}/*{}*".format(dir_datasets, name_plus_t))
        os.system("rm {}/*{}*".format(dir_datasets, name_plus_v))
        filenames_train = divide_to_multiple_csv_files(Data, Time_into_X, Time_into_Y, Time, sample_anchores_t, name_prefix_t, Nx, x_indices, Ny, y_indices, header=None, n_parts=nFiles_t)
        filenames_valid = divide_to_multiple_csv_files(Data, Time_into_X, Time_into_Y, Time, sample_anchores_v, name_prefix_v, Nx, x_indices, Ny, y_indices, header=None, n_parts=nFiles_v)

    size_x = get_timepoint_size(x_indices)
    size_y = get_timepoint_size(y_indices)

    # sample_anchores are already shuffled. But we need to shuffle datasets again, because it will reshuffle at every epoch.
    Dataset_train = csv_reader_to_dataset(filenames_train, Nx, size_x, Ny, size_y, Time_into_X, Time_into_Y, size_time,
                                n_parse_threads=5, batch_size=BatchSize, shuffle_buffer_size=BatchSize*shuffle_batch, n_readers=n_readers, transformer=Transformer)
    # Dataset_train = Dataset_train.prefetch(nPrefetch)

    Dataset_valid = csv_reader_to_dataset(filenames_valid, Nx, size_x, Ny, size_y, Time_into_X, Time_into_Y, size_time,
                                n_parse_threads=5, batch_size=BatchSize, shuffle_buffer_size=BatchSize*shuffle_batch, n_readers=n_readers, transformer=Transformer)
    # Dataset_valid = Dataset_valid.prefetch(nPrefetch)

    dx = size_x + (size_time if Time_into_X else 0)
    if Transformer: dx = dx + dx % 2
    dy = size_y + (size_time if Time_into_X else 0)     # not Time_into_Y here.
    if Transformer: dy = dy + dy % 2
    if Transformer: assert dx == dy

    return Dataset_train, Dataset_valid, dx, dy


class CandleDataset(tf.data.Dataset):
    def _generator(candles, nx, ny, time_into_x, x_indices, time_into_y, y_indices, times, sample_anchores):

        for anchor in sample_anchores:
            
            x = np.reshape(candles[anchor: anchor + nx][:, x_indices[0]][:, :, x_indices[1]], (nx, -1))
            if time_into_x is True:
                assert times is not None
                x_time = np.reshape(times[anchor: anchor + nx], (nx, -1))
                x = np.concatenate((x, x_time), axis=1)
           
            y = np.reshape(candles[anchor + nx: anchor + nx + ny][:, y_indices[0]][:, :, y_indices[1]], (ny, -1))
            if time_into_x is True:
                assert times is not None
                y_time = np.reshape(times[anchor + nx: anchor + nx + ny], (ny, -1))
                if time_into_y is True:
                    pass
                else:
                    y_time[:] = 0.0    # This is a placeholder for no-info. Losses and metrics will recognize it.
                y = np.concatenate((y, y_time), axis=1)

            x = tf.pad(x, [[1,1], [0,0]], constant_values=0)   # (1 pre-pad: Start, 1 post-pad: End) on axis 0. (0 pre-pad, 0 post-pad) on axis 1.
            y = tf.pad(y, [[1,1], [0,0]], constant_values=0)

            if x.shape[-1] % 2 != 0:
                x = tf.pad(x, [[0,0], [0,1]], constant_values=0) # (0 pre-pad: Start, 0 post-pad: End) on axis 0. (0 pre-pad, 1 post-pad) on axis 1.
                y = tf.pad(y, [[0,0], [0,1]], constant_values=0)

            yield ( ( tf.convert_to_tensor(x), tf.convert_to_tensor(y[:-1]) ), tf.convert_to_tensor(y[1:]) )
            # so M(x, [y[0]]) -> y[1], M(x, [y[0], y[1]]) -> y[2], ..., M(x, [y[0], ..., y[-2]]) -> y[-1]
            # where y[0] = Start, y[-1] = End.

    def get_timepoint_size(cls, indices):
        size = 1
        for ids in indices:
            size *= len(ids)
        return size

    def __new__(
            cls, 
            candles, time_into_x, time_into_y, times, sample_anchores,
            nx, x_indices, ny, y_indices, size_time
        ):
        sixe_x = get_timepoint_size(x_indices) + (size_time if time_into_x else 0)
        size_y = get_timepoint_size(y_indices) + (size_time if time_into_x else 0) # time_into_x
        
        return tf.data.Dataset.from_generator(
                cls._generator,
                output_signature = (
                    (
                        tf.TensorSpec(shape = (nx, sixe_x), dtype = candles.dtype),
                        tf.TensorSpec(shape = (ny, size_y), dtype = candles.dtype),
                    ),
                    tf.TensorSpec(shape = (ny, size_y), dtype = candles.dtype)
                ),
                args=(candles, nx, ny, time_into_x, x_indices, time_into_y, y_indices, times, sample_anchores)
        )

def anchor_to_sample_org(
    candles, nx, ny, anchor, x_indices, time_into_x, y_indices, time_into_y, times
):
    anchor = anchor.numpy()
    
    x = np.reshape(candles[anchor: anchor + nx][:, x_indices[0]][:, :, x_indices[1]], (nx, -1))
    # x = tf.reshape(candles[anchor: anchor + nx][:, x_indices[0]][:, :, x_indices[1]], (nx, -1))
    if time_into_x is True:
        assert times is not None
        x_time = np.reshape(times[anchor: anchor + nx], (nx, -1))
        # x_time = tf.reshape(times[anchor: anchor + nx], (nx, -1))
        x = np.concatenate((x, x_time), axis=1)
        # x = tf.concat((x, x_time), axis=1)

    y = np.reshape(candles[anchor + nx: anchor + nx + ny][:, y_indices[0]][:, :, y_indices[1]], (ny, -1))
    # y = tf.reshape(candles[anchor + nx: anchor + nx + ny][:, y_indices[0]][:, :, y_indices[1]], (ny, -1))
    if time_into_x is True:
        assert times is not None
        y_time = np.reshape(times[anchor + nx: anchor + nx + ny], (ny, -1))
        # y_time = tf.reshape(times[anchor + nx: anchor + nx + ny], (ny, -1))

        if time_into_y is True:
            pass
        else:
            y_time[:] = 0.0
            # y_time = tf.zeros_like(y_time)    # This is a placeholder for no-info. Losses and metrics will recognize it.
        y = np.concatenate((y, y_time), axis=1)
        # y = tf.concat((y, y_time), axis=1)

    x = np.pad(x, [[1,1], [0,0]], constant_values=0)   # (1 pre-pad: Start, 1 post-pad: End) on axis 0. (0 pre-pad, 0 post-pad) on axis 1.
    # x = tf.pad(x, [[1,1], [0,0]], constant_values=0)   # (1 pre-pad: Start, 1 post-pad: End) on axis 0. (0 pre-pad, 0 post-pad) on axis 1.
    y = np.pad(y, [[1,1], [0,0]], constant_values=0)
    # y = tf.pad(y, [[1,1], [0,0]], constant_values=0)

    if x.shape[-1] % 2 != 0:
        x = np.pad(x, [[0,0], [0,1]], constant_values=0) # (0 pre-pad: Start, 0 post-pad: End) on axis 0. (0 pre-pad, 1 post-pad) on axis 1.
        # x = tf.pad(x, [[0,0], [0,1]], constant_values=0) # (0 pre-pad: Start, 0 post-pad: End) on axis 0. (0 pre-pad, 1 post-pad) on axis 1.
        y = np.pad(y, [[0,0], [0,1]], constant_values=0)
        # y = tf.pad(y, [[0,0], [0,1]], constant_values=0)

    return (x, y[:-1], y[1:])
    # so M(x, [y[0]]) -> y[1], M(x, [y[0], y[1]]) -> y[2], ..., M(x, [y[0], ..., y[-2]]) -> y[-1]
    # where y[0] = Start, y[-1] = End.


def get_datasets_2(
    Candles, Time_into_X, Time_into_Y, Times, 
    sample_anchores_t, sample_anchores_v,
    Nx, x_indices, Ny, y_indices, size_time, target_markets,
    BatchSize, shuffle_batch, shuffle=True
):
    
    size_x = get_timepoint_size(x_indices)
    size_y = get_timepoint_size(y_indices) # time_into_x

    dx = size_x + (size_time if Time_into_X else 0)
    dx = dx + dx % 2
    dy = size_y + (size_time if Time_into_X else 0)     # not Time_into_Y
    dy = dy + dy % 2
    assert dx == dy


    def anchor_to_sample(anchor):

        x = np.reshape(Candles[anchor: anchor + Nx][:, x_indices[0]][:, :, x_indices[1]], (Nx, -1))
        if Time_into_X is True:
            assert Times is not None
            x_time = np.reshape(Times[anchor: anchor + Nx], (Nx, -1))
            # x_time = tf.reshape(times[anchor: anchor + nx], (nx, -1))
            x = np.concatenate((x, x_time), axis=1)
            # x = tf.concat((x, x_time), axis=1)

        y = np.reshape(Candles[anchor + Nx: anchor + Nx + Ny][:, y_indices[0]][:, :, y_indices[1]], (Ny, -1))

        if Time_into_X is True:
            assert Times is not None
            y_time = np.reshape(Times[anchor + Nx: anchor + Nx + Ny], (Ny, -1))
            if Time_into_Y is True:
                pass
            else:
                y_time[:] = 0.0
            y = np.concatenate((y, y_time), axis=1)

        x = np.pad(x, [[1,1], [0,0]], constant_values=0)   # (1 pre-pad: Start, 1 post-pad: End) on axis 0. (0 pre-pad, 0 post-pad) on axis 1.
        y = np.pad(y, [[1,1], [0,0]], constant_values=0)

        if x.shape[-1] % 2 != 0:
            x = np.pad(x, [[0,0], [0,1]], constant_values=0) # (0 pre-pad: Start, 0 post-pad: End) on axis 0. (0 pre-pad, 1 post-pad) on axis 1.
            y = np.pad(y, [[0,0], [0,1]], constant_values=0)

        return x, y[:-1], y_target[1:]
        # so M(x, [y[0]]) -> y[1], M(x, [y[0], y[1]]) -> y[2], ..., M(x, [y[0], ..., y[-2]]) -> y[-1]
        # where y[0] = Start, y[-1] = End.


    def refine_dataset(sample_anchores):
        dataset = tf.data.Dataset.from_tensor_slices(sample_anchores)
        dataset = dataset.map(
            lambda anchor: tf.numpy_function(
                # All are evaluated to numpy things. E.g. anchor is evaluated to numpy.
                anchor_to_sample,
                inp = [anchor],
                Tout = [Candles.dtype, Candles.dtype, Candles.dtype]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        ) \
        .map(lambda x, y, z: ((x, y), z))

        if shuffle is True:
            dataset = dataset.shuffle(BatchSize * shuffle_batch)
        
        dataset = dataset.batch(BatchSize, drop_remainder=False) \
        .prefetch(tf.data.AUTOTUNE)
        # .cache()

        # lambda anchor: 
        # tf.numpy_function(
        #     anchor_to_sample,
        #     inp = [Candles, Nx, Ny, anchor, x_indices, Time_into_X, y_indices, Time_into_Y, Times],
        #     Tout = [Candles.dtype, Candles.dtype, Candles.dtype])
            # Tout= [
            #     tf.TensorSpec(shape=[Nx, dx], dtype=Candles.dtype), 
            #     tf.TensorSpec(shape=[Ny, dy], dtype=Candles.dtype),
            #     tf.TensorSpec(shape=[Ny, dy], dtype=Candles.dtype)
            # ]
        # lambda anchor: anchor_to_sample(
        #     Candles, Nx, Ny, int(anchor), x_indices, Time_into_X, y_indices, Time_into_Y, Times
        # ),
        # num_parallel_calls=tf.data.AUTOTUNE

        # ds_valid = CandleDataset(
        #     Candles, Time_into_X, Time_into_Y, Times, sample_anchores_v,
        #     Nx, x_indices, Ny, y_indices, size_time
        # ) \
        # .shuffle(BatchSize * shuffle_batch) \
        # .batch(BatchSize, drop_remainder=False) \
        # .prefetch(tf.data.AUTOTUNE) \
        # .cache()

        return dataset

    ds_train = refine_dataset(sample_anchores_t)
    ds_valid = refine_dataset(sample_anchores_v)

    return ds_train, ds_valid, dx, dy


def build_model(
    dx, dy, Num_Layers, Num_Heads, Factor_FF, repComplexity, Dropout_Rate,
    HuberThreshold, CancleLossWeight, TrendLossWeight
    ):

    assert dx == dy

    cryptoformer = ConTransformer(
        num_layers=Num_Layers, d_model=dx, num_heads=Num_Heads, dff=Factor_FF*dx, 
        repComplexity=repComplexity, dropout_rate=Dropout_Rate
    )

    candle_input_x = keras.Input( shape=( (None, dx) ), name='candle_input_x' )
    candle_input_y = keras.Input( shape=( (None, dx) ), name='candle_input_y' )
    candle_input_shifted_y = keras.Input( shape=( (None, dx) ), name='candle_input_shifted_y' )

    cryptoformer_input = (candle_input_x, candle_input_y)
    cryptoformer_output = cryptoformer(cryptoformer_input)

    model = keras.Model(
        inputs = [cryptoformer_input],
        outputs = { "value": cryptoformer_output, "trend": cryptoformer_output }
    )

    learning_rate = CustomSchedule(dx)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            0.005, # learning_rate, # default is 0.001
            beta_1=0.9, 
            beta_2=0.98, 
            epsilon=1e-9
        ),
        loss={
            "value": MaskedHuber(threshold=HuberThreshold),
            "trend": MaskedTrendError() 
        },
        loss_weights={ 
            "value": CancleLossWeight, 
            "trend": TrendLossWeight 
        },
        metrics={ 
            "value": [MaskedHuber_Metric()], 
            "trend": [MaskedTrendError_Metric(), MaskedTrendAccuracy_Metric()]
        }
    )

    return model


def build_model_2(
    dx, dy, Num_Layers, Num_Heads, Factor_FF, repComplexity, Dropout_Rate,
    HuberThreshold, Optimizer, Learning_Rate
    ):

    assert dx == dy

    cryptoformer = ConTransformer(
        num_layers=Num_Layers, d_model=dx, num_heads=Num_Heads, dff=Factor_FF*dx, 
        repComplexity=repComplexity, dropout_rate=Dropout_Rate
    )

    candle_input_x = keras.Input( shape=( (None, dx) ), name='candle_input_x' )
    candle_input_y = keras.Input( shape=( (None, dx) ), name='candle_input_y' )
    candle_input_shifted_y = keras.Input( shape=( (None, dx) ), name='candle_input_shifted_y' )

    cryptoformer_input = (candle_input_x, candle_input_y)
    cryptoformer_output = cryptoformer(cryptoformer_input)

    model = keras.Model(
        inputs = [cryptoformer_input],
        outputs = [cryptoformer_output]
    )

    learning_rate = CustomSchedule(dx)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            Learning_Rate, # learning_rate, # default is 0.001. 
            beta_1=0.9,
            beta_2=0.98, 
            epsilon=1e-9
        ),
        loss=[MaskedHuber(threshold=HuberThreshold)],
        metrics=[MaskedTrendAccuracy_Metric()]
    )

    return model


def get_callbacks(
    checkpoint_filepath, Checkpoint_Monitor, 
    csvLogger_filepath, 
    EarlyStopping_Min_Monitor, EarlyStopping_Patience
):
    callbacks = []
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=Checkpoint_Monitor,
        mode='min',
        save_best_only=True,
        save_freq='epoch',
        # initial_value_threshold=0.5,
    )
    callbacks.append(model_checkpoint_callback)

    history_logger=tf.keras.callbacks.CSVLogger(csvLogger_filepath, separator=",", append=True)
    callbacks.append(history_logger)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor=EarlyStopping_Min_Monitor,
        patience=EarlyStopping_Patience,
        mode='min',
        restore_best_weights=True
    )
    callbacks.append(early_stopping_callback)

    return callbacks

#========================================= MaskedHuber

def MaskedHuber_Core(y_true, y_pred, sample_weight=None, threshold=1.0):
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
    loss = tf.divide(rs_masked_loss, rs_mask + 1e-9, name='loss')
    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, y_pred.dtype)
        sample_weight = tf.broadcast_to(sample_weight, loss.shape)
        loss = tf.multiply(loss, sample_weight)
    return loss


class MaskedHuber(tf.keras.losses.Loss):
    def __init__(self, name='mHuber', threshold=1.0, **kwargs):
        super(MaskedHuber, self).__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name, **kwargs)
        self.threshold = threshold
        
    def call(self, y_true, y_pred, sample_weight=None):
        loss = MaskedHuber_Core(y_true, y_pred, sample_weight)
        return loss


class MaskedHuber_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='mHuber', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_updates_seen = self.add_weight(name='num', initializer='zeros')
        self.avg_across_updates = self.add_weight(name='metric', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = loss = MaskedHuber_Core(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.reduce_mean(loss, axis=None)
        self.num_updates_seen.assign_add(tf.constant(1, dtype=self.dtype))
        self.avg_across_updates.assign_add( (loss - self.avg_across_updates) / (self.num_updates_seen + 1e-9) )

    def result(self):
        return self.avg_across_updates
    
    def reset_state(self):
        self.num_updates_seen.assign(0.)
        self.avg_across_updates.assign(0.)

#========================================= MaskedMSE

def MaskedMSE_Core(y_true, y_pred, sample_weight=None):
        # y_true, y_pred: (batch, sequence, depth)
        raw_loss = tf.square(y_true - y_pred)
        mask = tf.cast(y_true != 0, dtype=y_pred.dtype, name='mask')   # no need for 0.0
        masked_loss = tf.multiply(raw_loss, mask, name='masked_loss')
        rs_masked_loss = tf.reduce_sum(masked_loss, axis=-1, name='rs_masked_loss')  # depth axis is reduced
        rs_mask = tf.reduce_sum(mask, axis=-1, name='rs_mask')  # depth axis is reduced
        loss = tf.divide(rs_masked_loss, rs_mask + 1e-9, name='loss')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, y_pred.dtype)
            sample_weight = tf.broadcast_to(sample_weight, loss.shape)
            loss = tf.multiply(loss, sample_weight)
        return loss


class MaskedMSE(tf.keras.losses.Loss):
    def __init__(self, name='mMSE', **kwargs):
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name, **kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        loss = MaskedMSE_Core(y_true, y_pred, sample_weight=sample_weight)
        return loss


class MaskedMSE_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='mMSE', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_updates_seen = self.add_weight(name='num', initializer='zeros')
        self.avg_across_updates = self.add_weight(name='metric', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = MaskedMSE_Core(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.reduce_mean(loss, axis=None)
        self.num_updates_seen.assign_add(tf.constant(1, dtype=self.dtype))
        self.avg_across_updates.assign_add( (loss - self.avg_across_updates) / (self.num_updates_seen + 1e-9) )

    def result(self):
        return self.avg_across_updates
    
    def reset_state(self):
        self.num_updates_seen.assign(0.)
        self.avg_across_updates.assign(0.)

#========================================= MaskedTrendError

def MaskedTrendError_Core(y_true, y_pred, sample_weight=None):
    # y_true, y_pred: (batch, sequence, depth)
    d_true = (y_true[:, 1:, :] - y_true[:, :-1, :])
    d_true = d_true / tf.expand_dims(tf.norm(d_true, ord='euclidean', axis=-1) + 1e-30, axis=-1)
    d_pred = (y_pred[:, 1:, :] - y_pred[:, :-1, :])
    d_pred = d_pred / tf.expand_dims(tf.norm(d_pred, ord='euclidean', axis=-1) + 1e-30, axis=-1)
    mask =  tf.cast(y_true != 0, dtype=y_pred.dtype)   # no need for 0.0
    mask = tf.multiply(mask[:, 1:, :], mask[:, :-1, :], name='mask')
    raw_loss = - d_true * d_pred    # What's better?
    raw_loss = tf.nn.relu(raw_loss)
    masked_loss = tf.multiply(raw_loss, mask, name='masked_loss')
    rs_masked_loss = tf.reduce_sum(masked_loss, axis=-1, name='rs_masked_loss')  # depth axis is reduced
    rs_mask = tf.reduce_sum(mask, axis=-1, name='rs_mask')   # depth axis is reduced
    loss = tf.divide(rs_masked_loss, rs_mask + 1e-9, name='loss')  # (batch, sequence-1)
    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, y_pred.dtype)
        sample_weight = tf.broadcast_to(sample_weight, loss.shape)
        loss = tf.multiply(loss, sample_weight)
    return loss


class MaskedTrendError(tf.keras.losses.Loss):
    def __init__(self, name='mTE', **kwargs):
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name, **kwargs)
        
    def call(self, y_true, y_pred, sample_weight=None):
        loss = MaskedTrendError_Core(y_true, y_pred, sample_weight=sample_weight) # (batch, sequence-1)
        return loss


class MaskedTrendError_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='mTE', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_updates_seen = self.add_weight(name='num', initializer='zeros')
        self.avg_across_updates = self.add_weight(name='metric', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = MaskedTrendError_Core(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.reduce_mean(loss, axis=None)
        self.num_updates_seen.assign_add(1.)
        self.avg_across_updates.assign_add( (loss - self.avg_across_updates) / (self.num_updates_seen + 1e-9) )

    def result(self):
        return self.avg_across_updates
    
    def reset_state(self):
        self.num_updates_seen.assign(0.)
        self.avg_across_updates.assign(0.)


class MaskedTrendAccuracy_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='mTA', **kwargs):
        super().__init__(name=name, **kwargs)
        self.accMatch = self.add_weight(name='accMatch', initializer='zeros')
        self.accTotal = self.add_weight(name='accTotal', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask =  tf.cast(y_true != 0, dtype=y_pred.dtype)   # no need for 0.0
        mask = tf.multiply(mask[:, 1:, :], mask[:, :-1, :], name='mask')
        codir = (y_true[:, 1:, :] - y_true[:, :-1, :]) * (y_pred[:, 1:, :] - y_pred[:, :-1, :])
        masked_codir = tf.cast(tf.greater(codir, 0.0), y_pred.dtype) * mask
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, y_pred.dtype)
            sample_weight = tf.broadcast_to(sample_weight, masked_codir.shape)
            masked_codir = tf.multiply(masked_codir, sample_weight)
        self.accMatch.assign_add(tf.reduce_sum(masked_codir, axis=None))
        self.accTotal.assign_add(tf.reduce_sum(mask, axis=None))

    def result(self):
        return self.accMatch / (self.accTotal + 1e-30)
    
    def reset_state(self):
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


import pandas as pd
def plot_csv_log_history(file_path, columns):
    data = pd.read_csv(file_path)
    for col_name in columns:
        if col_name == 'epoch': continue
        plt.plot(range(len(data[col_name])), data[col_name], label=col_name)
    plt.legend(loc='center')
    plt.show()

# # Find market clusters # temporary
# from sklearn.metrics import pairwise

# distances = np.zeros( (Candles.shape[1], Candles.shape[1]), dtype=Candles.dtype)

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

