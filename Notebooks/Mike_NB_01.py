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


