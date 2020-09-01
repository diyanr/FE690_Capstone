from backtester.features.feature import Feature
import numpy as np
import pandas as pd
import hurstfunc as hf
from instruments import pairIds


class PlainMomentumPrediction(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()
        predictions = pd.Series(dtype='float64', index=instrumentManager.getAllInstrumentsByInstrumentId())

        # of upto lookback data points. The columns of this dataframe are the stock symbols/instrumentIds.
        mom10Data = lookbackInstrumentFeatures.getFeatureDf('mom_10')
        mom30Data = lookbackInstrumentFeatures.getFeatureDf('mom_30')
        ma90Data = lookbackInstrumentFeatures.getFeatureDf('ma_90')

        if len(ma90Data.index) > 30:
            mom10 = mom10Data.iloc[-1]
            mom30 = mom30Data.iloc[-1]
            # Go long if momentum is positive
            predictions[(mom30 > 0) & (mom10 > 0)] = 1
            # Go short if momentum is negative
            predictions[(mom30 <= 0) & (mom10 <= 0)] = 0
            # Get out of position if long term momentum is positive while short term is negative
            predictions[(mom30 > 0) & (mom10 <= 0)] = 0.5
            # Get out of position if long term momentum is negative while short term is positive
            predictions[(mom30 <= 0) & (mom10 > 0)] = 0.5
            print(f'mom10: {mom10.values}, mom30: {mom30.values}, pred: {predictions.values}')
        else:
            # If no sufficient data then don't take any positions
            predictions.values[:] = 0.5

        return predictions


class HurstMomentumPrediction(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()
        predictions = pd.Series(dtype='float64', index=instrumentManager.getAllInstrumentsByInstrumentId())

        # of upto lookback data points. The columns of this dataframe are the stock symbols/instrumentIds.
        mom10Data = lookbackInstrumentFeatures.getFeatureDf('mom_10')
        mom30Data = lookbackInstrumentFeatures.getFeatureDf('mom_30')
        ma90Data = lookbackInstrumentFeatures.getFeatureDf('ma_90')
        hc120Data = lookbackInstrumentFeatures.getFeatureDf('hc_120')

        if len(ma90Data.index) > 30:
            mom10 = mom10Data.iloc[-1]
            mom30 = mom30Data.iloc[-1]
            hc120 = hc120Data.iloc[-1]
            # Go long if Hurst > 0.5 and both long term and short term momentum are positive
            predictions[(hc120 > 0.5) & (mom30 > 0) & (mom10 > 0)] = 1
            # Go short if Hurst > 0.5 and both long term and short term momentum are negative
            predictions[(hc120 > 0.5) & (mom30 <= 0) & (mom10 <= 0)] = 0
            # Get out of position if Hurst > 0.5 and long term momentum is positive while short term is negative
            predictions[(hc120 > 0.5) & (mom30 > 0) & (mom10 <= 0)] = 0.5
            # Get out of position if Hurst > 0.5 and long term momentum is negative while short term is positive
            predictions[(hc120 > 0.5) & (mom30 <= 0) & (mom10 > 0)] = 0.5
            # Get out of position if Hurst < 0.5
            predictions[hc120 <= 0.5] = 0.5
            print(f'mom10: {mom10.values}, mom30: {mom30.values} , hc120: {hc120.values}')
        else:
            # If no sufficient data then don't take any positions
            predictions.values[:] = 0.5

        return predictions


class PlainPairValuePrediction(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        lookbackMarketFeatures = instrumentManager.getDataDf()

        prediction = pd.Series(0.5, index=instrumentManager.getAllInstrumentsByInstrumentId())
        if len(lookbackMarketFeatures) > 0:
            currentMarketFeatures = lookbackMarketFeatures.iloc[-1]
            z_score = pd.Series(dtype='float64', index=pairIds.keys())
            for i in pairIds.keys():
                z_score[i] = currentMarketFeatures[f'zscore_r{i}']

                # Go short on stock 1 and long on stock 2 if z-score > 1
                if z_score[i] > 1:
                    prediction[pairIds[i][0]] = 0
                    prediction[pairIds[i][1]] = 1
                # Go long on stock 1 and short on stock 2 if z-score < -1
                elif z_score[i] < -1:
                    prediction[pairIds[i][0]] = 1
                    prediction[pairIds[i][1]] = 0
                elif (z_score[i] > 0.5) or (z_score[i] < -0.5):
                    prediction[pairIds[i][0]] = 0.75
                    prediction[pairIds[i][1]] = 0.25
                # Don't take any position
                else:
                    prediction[pairIds[i][0]] = 0.5
                    prediction[pairIds[i][1]] = 0.5

        return prediction


class HurstPairValuePrediction(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        lookbackMarketFeatures = instrumentManager.getDataDf()

        prediction = pd.Series(0.5, index=instrumentManager.getAllInstrumentsByInstrumentId())
        if len(lookbackMarketFeatures) > 0:
            currentMarketFeatures = lookbackMarketFeatures.iloc[-1]
            z_score = pd.Series(dtype='float64', index=pairIds.keys())
            hc_120 = pd.Series(dtype='float64', index=pairIds.keys())
            for i in pairIds.keys():
                z_score[i] = currentMarketFeatures[f'zscore_r{i}']
                hc_120[i] = currentMarketFeatures[f'hurst_r{i}_120']
                # Go short on stock 1 and long on stock 2 if z-score > 1 and hurst < 0.5
                if z_score[i] > 1 and hc_120[i] < 0.5:
                    prediction[pairIds[i][0]] = 0
                    prediction[pairIds[i][1]] = 1
                # Go long on stock 1 and short on stock 2 if z-score < -1 and hurst < 0.5
                elif z_score[i] < -1 and hc_120[i] < 0.5:
                    prediction[pairIds[i][0]] = 1
                    prediction[pairIds[i][1]] = 0
                # Don't take any position
                else:
                    prediction[pairIds[i][0]] = 0.5
                    prediction[pairIds[i][1]] = 0.5

        return prediction


class HurstExponent(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        dataDf = instrumentLookbackData.getFeatureDf(featureParams['featureName'])
        instrumentDict = instrumentManager.getAllInstrumentsByInstrumentId()
        instrumentIds = list(instrumentDict.keys())

        if len(dataDf.index) > featureParams['period']:
            res = {}
            for instrumentId in instrumentIds:
                instData = dataDf[instrumentId].iloc[-featureParams['period']:]
                hurst_idx = hf.hurst_func(instData.values)
                res[instrumentId] = hurst_idx
            return pd.Series(res, index=instrumentIds)
        else:
            return pd.Series(0.5, index=instrumentIds)

    @classmethod
    def computeForMarket(cls, updateNum, time, featureParams, featureKey, currentMarketFeatures, instrumentManager):
        lookbackDataDf = instrumentManager.getDataDf()
        dataDf = lookbackDataDf[featureParams['featureName']]

        if len(dataDf) > featureParams['period']:
            marketData = dataDf.iloc[-featureParams['period']:]
            hurst_idx = hf.hurst_func(marketData.values)
            return hurst_idx
        else:
            return 0.5


class ZeeScore(Feature):
    @classmethod
    def computeForMarket(cls, updateNum, time, featureParams, featureKey, currentMarketFeatures, instrumentManager):
        i = featureParams['pairId']
        lookbackMarketFeatures = instrumentManager.getDataDf()
        z_score = 0.0
        if len(lookbackMarketFeatures) > 0:
            currentMarketFeatures = lookbackMarketFeatures.iloc[-1]
            if currentMarketFeatures[f'sdev_r{i}_90'] != 0:
                z_score = ((currentMarketFeatures[f'ma_r{i}_10']
                            - currentMarketFeatures[f'ma_r{i}_90'])
                           / currentMarketFeatures[f'sdev_r{i}_90'])
            else:
                z_score = 0.0

            if currentMarketFeatures[f'correl_r{i}_90'] < 0.5:
                z_score = 0.0

        return z_score


class FeesCalculator(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        return pd.Series(0, index=instrumentManager.getAllInstrumentsByInstrumentId())


class BuyHoldPnL(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()

        priceData = instrumentLookbackData.getFeatureDf('adjClose')

        if len(priceData) < 2:
            return pd.Series(0, index=instrumentManager.getAllInstrumentsByInstrumentId())
        else:
            bhpnl = instrumentLookbackData.getFeatureDf(featureKey).iloc[-1]
            bhpnl += priceData.iloc[-1] - priceData.iloc[-2]

        return bhpnl


class ScoreFeature(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        if len(instrumentLookbackData.getFeatureDf(featureParams['featureName1'])) > 0:
            feature1 = instrumentLookbackData.getFeatureDf(featureParams['featureName1']).iloc[-1]
            feature2 = instrumentLookbackData.getFeatureDf(featureParams['featureName2']).iloc[-1]

            for instrumentId in feature1.index:
                pnls = instrumentLookbackData.getFeatureDf('pnl')[instrumentId]
                positions = instrumentLookbackData.getFeatureDf('position')[instrumentId]

                # print(instrumentId)
                print('pnl: %.2f' % pnls[-1])
                if len(positions) > 2 and np.abs(positions[-1] - positions[-2]) > 0:
                    print('Position changed to: %.2f' % positions[-1])

            toRtn = (feature1 - feature2) / feature2.abs()
            toRtn[toRtn.isnull()] = 0
            toRtn[toRtn == np.Inf] = 0
        else:
            toRtn = 0

        return toRtn
