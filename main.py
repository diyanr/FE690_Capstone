from backtester.trading_system import TradingSystem

from momentum_hurst import HurstMomParams
from momentum_plain import PlainMomParams
from pairs_hurst import HurstPairParams
from pairs_plain import PlainPairParams


def run_plain_momentum():
    tsParams = PlainMomParams()
    tradingSystem = TradingSystem(tsParams)
    results = tradingSystem.startTrading()
    return results


def run_hurst_momentum():
    tsParams = HurstMomParams()
    tradingSystem = TradingSystem(tsParams)
    results = tradingSystem.startTrading()
    return results


def run_plain_pairs():
    tsParams = PlainPairParams()
    tradingSystem = TradingSystem(tsParams)
    result = tradingSystem.startTrading()
    return result


def run_hurst_pairs():
    tsParams = HurstPairParams()
    tradingSystem = TradingSystem(tsParams)
    result = tradingSystem.startTrading()
    return result


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(run_hurst_momentum())
