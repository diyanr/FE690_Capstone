import momentum as mom
from backtester.trading_system import TradingSystem


def run_momentum():
    tf = mom.MyTradingFunctions()
    tsParams = mom.MyTradingParams(tf)
    tradingSystem = TradingSystem(tsParams)
    results = tradingSystem.startTrading()
    return results


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(run_momentum)

