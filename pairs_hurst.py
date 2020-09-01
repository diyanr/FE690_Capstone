from backtester.constants import *
from backtester.dataSource.yahoo_data_source import YahooStockDataSource
from backtester.executionSystem.simple_execution_system import SimpleExecutionSystem
from backtester.orderPlacer.backtesting_order_placer import BacktestingOrderPlacer
from backtester.timeRule.custom_time_rule import CustomTimeRule
from backtester.trading_system import TradingSystem
from backtester.trading_system_parameters import TradingSystemParameters

from features import HurstPairValuePrediction, ZeeScore, HurstExponent
from instruments import pairIds, pairInstIds, pairStartDate, pairEndDate


class HurstPairParams(TradingSystemParameters):
    def __init__(self):
        super(HurstPairParams, self).__init__()
        self.__dataSetId = 'testPairsTrading'
        self.__instrumentIds = pairInstIds
        self.__startDate = pairStartDate
        self.__endDate = pairEndDate

    def getDataParser(self):
        return YahooStockDataSource(cachedFolderName='yahooData/',
                                    dataSetId='testPairsTrading',
                                    instrumentIds=self.__instrumentIds,
                                    startDateStr=self.__startDate,
                                    endDateStr=self.__endDate)

    def getBenchmark(self):
        return None

    def getCustomFeatures(self):
        """
        This is a way to use any custom features you might have made.
        Returns a dictionary where
        key: featureId to access this feature (Make sure this doesnt conflict with any of the pre defined feature Ids)
        value: Your custom Class which computes this feature. The class should be an instance of Feature
        Eg. if your custom class is MyCustomFeature, and you want to access this via featureId='my_custom_feature',
        you will import that class, and return this function as {'my_custom_feature': MyCustomFeature}
        """
        return {
            'hurst_pairs_prediction': HurstPairValuePrediction,
            'zscore': ZeeScore,
            'hurst': HurstExponent
        }

    def getTimeRuleForUpdates(self):
        return CustomTimeRule(
            startDate=self.__startDate,
            endDate=self.__endDate,
            frequency='D',
            sample='30'
        )

    # def getTimeRuleForUpdates(self):
    #     return USTimeRule(startDate=self.__startDate,
    #                       endDate=self.__endDate)

    def getInstrumentsIds(self):
        """
        Get all instrument ids
        """

        return self.__instrumentIds

    # def getStartingCapital(self):
    #     if len(self.getInstrumentsIds()) > 0:
    #         return 1000 * len(self.getInstrumentsIds())
    #     else:
    #         return 30000

    def getPairIds(self):
        return self.__pairIds

    def getInstrumentFeatureConfigDicts(self):
        ma1Dict = {'featureKey': 'ma_90',
                   'featureId': 'moving_average',
                   'params': {'period': 90,
                              'featureName': 'adjClose'}}
        ma2Dict = {'featureKey': 'ma_15',
                   'featureId': 'moving_average',
                   'params': {'period': 15,
                              'featureName': 'adjClose'}}
        sdevDict = {'featureKey': 'ma_15',
                    'featureId': 'moving_average',
                    'params': {'period': 15,
                               'featureName': 'adjClose'}}
        pairValuePrediction = {'featureKey': 'prediction',
                               'featureId': 'hurst_pairs_prediction',
                               'params': {}}
        return {INSTRUMENT_TYPE_STOCK: [pairValuePrediction]}

    def getMarketFeatureConfigDicts(self):
        """
        Returns an array of market feature config dictionaries
            market feature config Dictionary has the following keys:
            featureId: a string representing the type of feature you want to use
            featureKey: a string representing the key you will use to access the value of this feature.this
            params: A dictionary with which contains other optional params if needed by the feature
        """
        ratio1Dict = {'featureKey': 'ratio1',
                      'featureId': 'ratio',
                      'params': {'instrumentId1': pairIds[1][0],
                                 'instrumentId2': pairIds[1][1],
                                 'featureName': 'adjClose'}}
        ma11Dict = {'featureKey': 'ma_r1_90',
                    'featureId': 'moving_average',
                    'params': {'period': 90,
                               'featureName': 'ratio1'}}
        ma21Dict = {'featureKey': 'ma_r1_10',
                    'featureId': 'moving_average',
                    'params': {'period': 10,
                               'featureName': 'ratio1'}}
        sdev1Dict = {'featureKey': 'sdev_r1_90',
                     'featureId': 'moving_sdev',
                     'params': {'period': 90,
                                'featureName': 'ratio1'}}
        correl1Dict = {'featureKey': 'correl_r1_90',
                       'featureId': 'cross_instrument_correlation',
                       'params': {'period': 90,
                                  'instrumentId1': pairIds[1][0],
                                  'instrumentId2': pairIds[1][1],
                                  'featureName': 'adjClose'}}
        zscore1Dict = {'featureKey': 'zscore_r1',
                       'featureId': 'zscore',
                       'params': {'pairId': list(pairIds.keys())[0]}}
        hurst1Dict = {'featureKey': 'hurst_r1_120',
                      'featureId': 'hurst',
                      'params': {'period': 120,
                                 'featureName': 'ratio1'}}
        ratio2Dict = {'featureKey': 'ratio2',
                      'featureId': 'ratio',
                      'params': {'instrumentId1': pairIds[2][0],
                                 'instrumentId2': pairIds[2][1],
                                 'featureName': 'adjClose'}}
        ma12Dict = {'featureKey': 'ma_r2_90',
                    'featureId': 'moving_average',
                    'params': {'period': 90,
                               'featureName': 'ratio2'}}
        ma22Dict = {'featureKey': 'ma_r2_10',
                    'featureId': 'moving_average',
                    'params': {'period': 10,
                               'featureName': 'ratio2'}}
        sdev2Dict = {'featureKey': 'sdev_r2_90',
                     'featureId': 'moving_sdev',
                     'params': {'period': 90,
                                'featureName': 'ratio2'}}
        correl2Dict = {'featureKey': 'correl_r2_90',
                       'featureId': 'cross_instrument_correlation',
                       'params': {'period': 90,
                                  'instrumentId1': pairIds[2][0],
                                  'instrumentId2': pairIds[2][1],
                                  'featureName': 'adjClose'}}
        zscore2Dict = {'featureKey': 'zscore_r2',
                       'featureId': 'zscore',
                       'params': {'pairId': list(pairIds.keys())[1]}}
        hurst2Dict = {'featureKey': 'hurst_r2_120',
                      'featureId': 'hurst',
                      'params': {'period': 120,
                                 'featureName': 'ratio2'}}

        return [ratio1Dict, ma11Dict, ma21Dict, sdev1Dict, correl1Dict, zscore1Dict, hurst1Dict,
                ratio2Dict, ma12Dict, ma22Dict, sdev2Dict, correl2Dict, zscore2Dict, hurst2Dict]

    def getExecutionSystem(self):
        """
        Returns the type of execution system we want to use. Its an implementation of the class ExecutionSystem
        It converts prediction to intended positions for different instruments.
        """
        return SimpleExecutionSystem(enter_threshold=0.7,
                                     exit_threshold=0.55,
                                     longLimit=1,
                                     shortLimit=1,
                                     capitalUsageLimit=0.85,
                                     enterlotSize=1,
                                     exitlotSize=1,
                                     limitType='L',
                                     price='adjClose'
        )

        # return SimpleExecutionSystem(enter_threshold=0.7,
        #                              exit_threshold=0.55,
        #                              longLimit=10000,
        #                              shortLimit=10000,
        #                              capitalUsageLimit=.85,
        #                              enterlotSize=10000,
        #                              exitlotSize=10000,
        #                              limitType='D',
        #                              price='adjClose')

    def getOrderPlacer(self):
        """
        Returns the type of order placer we want to use. its an implementation of the class OrderPlacer.
        It helps place an order, and also read confirmations of orders being placed.
        For Backtesting, you can just use the BacktestingOrderPlacer, which places the order which you want, and automatically confirms it too.
        """
        return BacktestingOrderPlacer()

    def getLookbackSize(self):
        """
        Returns the amount of lookback data you want for your calculations. The historical market features and instrument features are only
        stored upto this amount.
        This number is the number of times we have updated our features.
        """
        return 150


def run_hurst_pairs():
    tsParams = HurstPairParams()
    tradingSystem = TradingSystem(tsParams)
    result = tradingSystem.startTrading()
    return result


if __name__ == "__main__":
    print(run_hurst_pairs())
