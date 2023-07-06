
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class IterativeBase():
    """
    Base class for iterative (event-driven) backtesting of trading strategies.
    """
    
    def __init__(self, symbol, start, end, amount, data, charge = 0.00025):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        amount: float
            initial amount to be invested per trade
        data: str
            The location of dataset
        charge: charge*amount (default = 0.00025) 
            trading cost

    '''
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = amount
        self.current_balance = amount
        self.data_link = data
        self.units = 0
        self.trades = 0
        self.position = 0
        self.charge = charge
        self.get_data()
        
    def get_data(self):
        ''' Imports the data from data source(source can be changed).
        '''
        raw = pd.read_csv(self.data_link, parse_dates = ['Date'], index_col = "Date").dropna()
        raw = raw.loc[self.start:self.end].copy()
        raw["returns"] = np.log(raw.Close / raw.Close.shift(1))
        self.data = raw
        
    def plot_data(self, cols = None):  
        ''' Plots the closing price for the symbol.
        '''
        if cols is None:
            cols = "Close"
        self.data[cols].plot(figsize = (12, 8), title = self.symbol)
        
    def get_values(self, bar):
        ''' Returns the date, the price and the spread for the given bar.
        '''
        date = str(self.data.index[bar].date())
        price = round(self.data.Close.iloc[bar], 5)
        return date, price
    
    def print_current_balance(self, bar):
        ''' Prints out the current (cash) balance.
        '''
        date, price = self.get_values(bar)
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))
        
    def buy_instrument(self, bar, units = None, amount = None):
        ''' Places and executes a buy order (market order).
        '''
        if self.current_balance <= 0:
            return
        date, price = self.get_values(bar)
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        if self.charge:
            total_price = units*price*(1+self.charge)
            
        if total_price >= self.current_balance:
            while total_price >= self.current_balance:
                units -= 1
                total_price = units*price*(1+self.charge)
                if units <= 0:
                    return
        if units <= 0:
            return
        self.current_balance -= total_price # reduce cash balance by "purchase price"
        self.units += units
        self.trades += 1
        print("{} |  Buying {} for {}".format(date, units, round(price, 5)))
        
    def sell_instrument(self, bar, units = None, amount = None):
        ''' Places and executes a sell order (market order).
        '''
        date, price = self.get_values(bar)
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        if self.charge:
            total_price = units*price*(1-self.charge)
        if self.units <=0 or units<=0:
            return
        self.current_balance += total_price # increases cash balance by "purchase price"
        self.units -= units
        self.trades += 1
        print("{} |  Selling {} for {}".format(date, units, round(price, 5)))
        
    def print_current_position_value(self, bar):
        ''' Prints out the current position value.
        '''
        date, price = self.get_values(bar)
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))
        
    def print_current_nav(self, bar):
        ''' Prints out the current net asset value (nav).
        '''
        date, price = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))
        
    def close_pos(self, bar):
        ''' Closes out a long or short position (go neutral).
        '''
        date, price = self.get_values(bar)
        print(75 * "-")
        print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price # closing final position (works with short and long!)
        self.current_balance -= self.units * price * self.charge # substract charge
        print("{} | closing position of {} for {}".format(date, self.units, price))
        self.units = 0 # setting position to neutral
        self.trades += 1
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        self.print_current_balance(bar)
        print("{} | net performance (%) = {}".format(date, round(perf, 2) ))
        print("{} | number of trades executed = {}".format(date, self.trades))
        print(75 * "-")