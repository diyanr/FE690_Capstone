from backtester.trading_system_parameters import TradingSystemParameters
from backtester.features.feature import Feature
from backtester.dataSource.yahoo_data_source import YahooStockDataSource
from backtester.timeRule.custom_time_rule import CustomTimeRule
from backtester.executionSystem.simple_execution_system import SimpleExecutionSystem
from backtester.orderPlacer.backtesting_order_placer import BacktestingOrderPlacer
from backtester.trading_system import TradingSystem
from backtester.constants import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta


class MyTradingParams(TradingSystemParameters):

    def __init__(self, tradingFunctions):
        self.__tradingFunctions = tradingFunctions
        super(MyTradingParams, self).__init__()
        self.__dataSetId = 'equity_data'
        self.__instrumentIds = self.__tradingFunctions.getSymbolsToTrade()
        self.__startDate = '2015/01/02'
        self.__endDate = '2017/08/31'

    def getDataParser(self):
        """
        Returns an instance of class DataParser. Source of data for instruments
        """
        instrumentIds = self.__tradingFunctions.getSymbolsToTrade()
        return YahooStockDataSource(
            cachedFolderName='historicalData/',
            dataSetId=self.__dataSetId,
            instrumentIds=instrumentIds,
            startDateStr=self.__startDate,
            endDateStr=self.__endDate,
        )

    def getTimeRuleForUpdates(self):
        return CustomTimeRule(
            startDate=self.__startDate,
            endDate=self.__endDate,
            frequency='D',
            sample='30'
        )

    def getFrequencyOfFeatureUpdates(self):
        return timedelta(days=1)

    def getStartingCapital(self):
        if len(self.__tradingFunctions.getSymbolsToTrade()) > 0:
            return 1000 * len(self.__tradingFunctions.getSymbolsToTrade())
        else:
            return 30000

    def getCustomFeatures(self):
        """
        This is a way to use any custom features you might have made.
        Returns a dictionary where:

        key: featureId to access this feature (Make sure this doesnt conflict with any of the pre defined feature Ids)
        value: Your custom Class which computes this feature. The class should be an instance of Feature

        Eg. if your custom class is MyCustomFeature, and you want to access this via featureId='my_custom_feature',
        you will import that class, and return this function as {'my_custom_feature': MyCustomFeature}
        """
        return {
            'my_custom_feature': MyCustomFeature,
            'prediction': TrainingPredictionFeature,
            'zero_fees': FeesCalculator,
            'benchmark_PnL': BuyHoldPnL,
            'score': ScoreFeature
        }

    def getInstrumentFeatureConfigDicts(self):
        """
        Returns an array of instrument feature config dictionaries instrument feature config Dictionary has the
        following keys:

        featureId: a string representing the type of feature you want to use
        featureKey: a string representing the key you will use to access the value of this feature
        params: A dictionary with which contains other optional params if needed by the feature
        """

        predictionDict = {
            'featureKey': 'prediction',
            'featureId': 'prediction',
            'params': {}
        }
        feesConfigDict = {
            'featureKey': 'fees',
            'featureId': 'zero_fees',
            'params': {}
        }
        profitlossConfigDict = {
            'featureKey': 'pnl',
            'featureId': 'pnl',
            'params': {
                'price': self.getPriceFeatureKey(),
                'fees': 'fees'
            }
        }
        capitalConfigDict = {
            'featureKey': 'capital',
            'featureId': 'capital',
            'params': {
                'price': 'adjClose',
                'fees': 'fees',
                'capitalReqPercent': 0.95
            }
        }
        benchmarkDict = {
            'featureKey': 'benchmark',
            'featureId': 'benchmark_PnL',
            'params': {'pnlKey': 'pnl'}
        }
        scoreDict = {
            'featureKey': 'score',
            'featureId': 'score',
            'params': {
                'featureName1': 'pnl',
                'featureName2': 'benchmark'
            }
        }

        stockFeatureConfigs = self.__tradingFunctions.getInstrumentFeatureConfigDicts()

        return {
            INSTRUMENT_TYPE_STOCK: stockFeatureConfigs + [
                predictionDict,
                feesConfigDict,
                profitlossConfigDict,
                capitalConfigDict,
                benchmarkDict,
                scoreDict
            ]
        }

    def getMarketFeatureConfigDicts(self):
        """
        Returns an array of market feature config dictionaries having the following keys:

        featureId: a string representing the type of feature you want to use
        featureKey: a string representing the key you will use to access the value of this feature
        params: A dictionary with which contains other optional params if needed by the feature
        """
        scoreDict = {
            'featureKey': 'score',
            'featureId': 'score_ll',
            'params': {
                'featureName': self.getPriceFeatureKey(),
                'instrument_score_feature': 'score'
            }
        }

        return [scoreDict]

    def getPrediction(self, time, updateNum, instrumentManager):
        predictions = pd.Series(dtype='float64', index=self.__instrumentIds)
        predictions = self.__tradingFunctions.getPrediction(time, updateNum, instrumentManager, predictions)

        return predictions

    def getExecutionSystem(self):
        """
        Returns the type of execution system we want to use. Its an implementation of the class ExecutionSystem
        It converts prediction to intended positions for different instruments.
        """

        return SimpleExecutionSystem(
            enter_threshold=0.7,
            exit_threshold=0.55,
            longLimit=1,
            shortLimit=1,
            capitalUsageLimit=0.10 * self.getStartingCapital(),
            enterlotSize=1,
            exitlotSize=1,
            limitType='L',
            price='adjClose'
        )

    def getOrderPlacer(self):
        """
        Returns the type of order placer we want to use. It's an implementation of the class OrderPlacer.
        It helps place an order, and also read confirmations of orders being placed.
        For Backtesting, you can just use the BacktestingOrderPlacer, which places the order which you want, and
        automatically confirms it too.
        """

        return BacktestingOrderPlacer()

    def getLookbackSize(self):
        """
        Returns the amount of lookback data you want for your calculations. The historical market features and instrument features are only
        stored upto this amount.
        This number is the number of times we have updated our features.
        """

        return 120

    def getPriceFeatureKey(self):
        """
        The name of column containing the instrument price
        """

        return 'adjClose'

    def getInstrumentsIds(self):
        """
        Get all instrument ids
        """

        return self.__instrumentIds


class TrainingPredictionFeature(Feature):

    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        tf = MyTradingFunctions()
        t = MyTradingParams(tf)

        return t.getPrediction(time, updateNum, instrumentManager)


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

                print(instrumentId)
                print('pnl: %.2f' % pnls[-1])
                if len(positions) > 2 and np.abs(positions[-1] - positions[-2]) > 0:
                    print('Position changed to: %.2f' % positions[-1])

            toRtn = (feature1 - feature2) / feature2.abs()
            toRtn[toRtn.isnull()] = 0
            toRtn[toRtn == np.Inf] = 0
        else:
            toRtn = 0

        return toRtn


class MyTradingFunctions():

    def __init__(self):
        self.count = 0
        self.params = {}

    def getSymbolsToTrade(self):
        """
        Specify the stock names that you want to trade.
        """

        return ['AAPL']

    def getInstrumentFeatureConfigDicts(self):
        """
        Specify all Features you want to use by creating config dictionaries.
        Create one dictionary per feature and return them in an array.

        Feature config Dictionary have the following keys:

        featureId: a str for the type of feature you want to use
        featureKey: {optional} a str for the key you will use to call this feature
                    If not present, will just use featureId
        params: {optional} A dictionary with which contains other optional params if needed by the feature

        msDict = {
            'featureKey': 'ms_5',
            'featureId': 'moving_sum',
            'params': {
                'period': 5,
                'featureName': 'basis'
            }
        }

        return [msDict]

        You can now use this feature by in getPRediction() calling it's featureKey, 'ms_5'
        """

        ma1Dict = {
            'featureKey': 'ma_90',
            'featureId': 'moving_average',
            'params': {
                'period': 90,
                'featureName': 'adjClose'
            }
        }
        mom30Dict = {
            'featureKey': 'mom_30',
            'featureId': 'momentum',
            'params': {
                'period': 30,
                'featureName': 'adjClose'
            }
        }
        mom10Dict = {
            'featureKey': 'mom_10',
            'featureId': 'momentum',
            'params': {
                'period': 10,
                'featureName': 'adjClose'
            }
        }

        return [ma1Dict, mom10Dict, mom30Dict]

    def getPrediction(self, time, updateNum, instrumentManager, predictions):
        """
        Combine all the features to create the desired predictions for each stock.
        'predictions' is Pandas Series with stock as index and predictions as values
        We first call the holder for all the instrument features for all stocks as
            lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()
        Then call the dataframe for a feature using its feature_key as
            ms5Data = lookbackInstrumentFeatures.getFeatureDf('ms_5')
        This returns a dataFrame for that feature for ALL stocks for all times upto lookback time
        Now you can call just the last data point for ALL stocks as
            ms5 = ms5Data.iloc[-1]
        You can call last datapoint for one stock 'ABC' as
            value_for_abs = ms5['ABC']

        Output of the prediction function is used by the toolbox to make further trading decisions and evaluate your score.
        """

        # self.updateCount() - uncomment if you want a counter

        # holder for all the instrument features for all instruments
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()

        # of upto lookback data points. The columns of this dataframe are the stock symbols/instrumentIds.
        mom30Data = lookbackInstrumentFeatures.getFeatureDf('mom_30')
        ma90Data = lookbackInstrumentFeatures.getFeatureDf('ma_90')

        if len(ma90Data.index) > 20:
            mom30 = mom30Data.iloc[-1]
            # Go long if momentum is positive
            predictions[mom30 > 0] = 1
            # Go short if momentum is negative
            predictions[mom30 <= 0] = 0
        else:
            # If no sufficient data then don't take any positions
            predictions.values[:] = 0.5
        return predictions

    def updateCount(self):
        self.count = self.count + 1


class MyCustomFeature(Feature):
    """
    Custom Feature to implement for instrument. This function would return the value of the feature you want to implement.
    1. create a new class MyCustomFeatureClassName for the feature and implement your logic in the function computeForInstrument() -

    2. modify function getCustomFeatures() to return a dictionary with Id for this class
        (follow formats like {'my_custom_feature_identifier': MyCustomFeatureClassName}.
        Make sure 'my_custom_feature_identifier' doesnt conflict with any of the pre defined feature Ids

        def getCustomFeatures(self):
            return {'my_custom_feature_identifier': MyCustomFeatureClassName}

    3. create a dict for this feature in getInstrumentFeatureConfigDicts() above. Dict format is:
            customFeatureDict = {'featureKey': 'my_custom_feature_key',
                                'featureId': 'my_custom_feature_identifier',
                                'params': {'param1': 'value1'}}
    You can now use this feature by calling it's featureKey, 'my_custom_feature_key' in getPrediction()
    """

    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        # Custom parameter which can be used as input to computation of this feature
        param1Value = featureParams['param1']

        # A holder for the all the instrument features
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()

        # dataframe for a historical instrument feature (basis in this case). The index is the timestamps
        # atmost upto lookback data points. The columns of this dataframe are the stocks/instrumentIds.
        lookbackInstrumentValue = lookbackInstrumentFeatures.getFeatureDf('adjClose')

        # The last row of the previous dataframe gives the last calculated value for that feature (basis in this case)
        # This returns a series with stocks/instrumentIds as the index.
        currentValue = lookbackInstrumentValue.iloc[-1]

        if param1Value == 'value1':
            return currentValue * 0.1
        else:
            return currentValue * 0.5


def run_plain_momentum():
    tf = MyTradingFunctions()
    tsParams = MyTradingParams(tf)
    tradingSystem = TradingSystem(tsParams)
    results = tradingSystem.startTrading()
    return results


if __name__ == '__main__':
    print(run_plain_momentum())
