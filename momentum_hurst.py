from datetime import timedelta

from backtester.constants import *
from backtester.dataSource.yahoo_data_source import YahooStockDataSource
from backtester.executionSystem.simple_execution_system import SimpleExecutionSystem
from backtester.orderPlacer.backtesting_order_placer import BacktestingOrderPlacer
from backtester.timeRule.custom_time_rule import CustomTimeRule
from backtester.trading_system import TradingSystem
from backtester.trading_system_parameters import TradingSystemParameters

import instruments as ins
from features import HurstMomentumPrediction, FeesCalculator, BuyHoldPnL, ScoreFeature, HurstExponent


class HurstMomParams(TradingSystemParameters):

    def __init__(self):
        super(HurstMomParams, self).__init__()
        self.__dataSetId = 'equity_data'
        self.__instrumentIds = ins.momInstIds
        self.__startDate = ins.momStartDate
        self.__endDate = ins.momEndDate

    def getDataParser(self):
        """
        Returns an instance of class DataParser. Source of data for instruments
        """
        return YahooStockDataSource(
            cachedFolderName='historicalData/',
            dataSetId=self.__dataSetId,
            instrumentIds=self.__instrumentIds,
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
        if len(self.getInstrumentsIds()) > 0:
            return 1000 * len(self.getInstrumentsIds())
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
            'hurst_momentum_prediction': HurstMomentumPrediction,
            'zero_fees': FeesCalculator,
            'benchmark_PnL': BuyHoldPnL,
            'score': ScoreFeature,
            'hurst': HurstExponent
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
            'featureId': 'hurst_momentum_prediction',
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
        hcDict = {
            'featureKey': 'hc_120',
            'featureId': 'hurst',
            'params': {
                'period': 120,
                'featureName': 'adjClose'
            }
        }
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

        return {
            INSTRUMENT_TYPE_STOCK: [ma1Dict,
                                    mom10Dict,
                                    mom30Dict,
                                    predictionDict,
                                    feesConfigDict,
                                    profitlossConfigDict,
                                    capitalConfigDict,
                                    benchmarkDict,
                                    scoreDict,
                                    hcDict]
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

    # def getPrediction(self, time, updateNum, instrumentManager):
    #     predictions = pd.Series(dtype='float64', index=self.__instrumentIds)
    #     predictions = self.__tradingFunctions.getPrediction(time, updateNum, instrumentManager, predictions)
    #
    #     return predictions

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
        '''
        Returns the type of order placer we want to use. It's an implementation of the class OrderPlacer.
        It helps place an order, and also read confirmations of orders being placed.
        For Backtesting, you can just use the BacktestingOrderPlacer, which places the order which you want, and
        automatically confirms it too.
        '''

        return BacktestingOrderPlacer()

    def getLookbackSize(self):
        """
        Returns the amount of lookback data you want for your calculations. The historical market features and instrument features are only
        stored upto this amount.
        This number is the number of times we have updated our features.
        """

        return 150

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


def run_hurst_momentum():
    tsParams = HurstMomParams()
    tradingSystem = TradingSystem(tsParams)
    results = tradingSystem.startTrading()
    return results


if __name__ == '__main__':
    print(run_hurst_momentum())
