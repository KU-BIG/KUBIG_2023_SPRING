import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from Iterative_Backtester import IterativeBase

class IterativeBacktest(IterativeBase):
    def go_long(self, bar, units = None, amount = None):
        if units:
            self.buy_instrument(bar, units = units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.buy_instrument(bar, amount = amount)
            
    def go_short(self, bar, units = None, amount = None):
        if units:
            self.sell_instrument(bar, units = units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.sell_instrument(bar, amount = amount) # go short
            
    def SMA_BOL(self, SMA_S, SMA_M, SMA_L, dev):
        ''' 
        Backtests an SMA crossover strategy with SMA_S (short) and SMA_L (long).
        
        Parameters
        ----------
        SMA_S: int
            moving window in bars (e.g. days) for shorter SMA
        SMA_L: int
            moving window in bars (e.g. days) for longer SMA
        '''
        stm = "Testing SMA With Bolinger strategy | {} | SMA_S = {} & SMA_M = {} & SMA_L = {} DEV = {}".format(self.symbol, SMA_S, SMA_M, SMA_L, dev)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data() # reset dataset
        
        self.data["SMA_S"] = self.data["Close"].rolling(SMA_S).mean()
        self.data["SMA_L"] = self.data["Close"].rolling(SMA_L).mean()
        
        self.data["SMA"] = self.data["Close"].rolling(SMA_M).mean()
        self.data["Lower_1"] = self.data["SMA"] - self.data["Close"].rolling(SMA_M).std() * (dev/2)
        self.data["Upper_1"] = self.data["SMA"] + self.data["Close"].rolling(SMA_M).std() * (dev/2)
        self.data["Lower_2"] = self.data["SMA"] - self.data["Close"].rolling(SMA_M).std() * dev
        self.data["Upper_2"] = self.data["SMA"] + self.data["Close"].rolling(SMA_M).std() * dev     

        self.data.dropna(inplace = True)
        
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            trade_flag = self.trades
            if self.data["SMA_S"].iloc[bar] > self.data["SMA_L"].iloc[bar]: # signal to go long
                if self.position in [0,-1]:
                    self.position = 1  # long position
            elif self.data["SMA_S"].iloc[bar] < self.data["SMA_L"].iloc[bar]: # signal to go short
                if self.position in [0, 1]:
                    self.position = -1 # short position
            else:
                if self.position in [1,-1]:
                    self.position = 0 #neutral position
            if self.position == 1 and self.data['Close'].iloc[bar]<self.data['Lower_2'].iloc[bar]:
                self.go_long(bar, amount='all')
            elif self.position == 1 and self.data['Close'].iloc[bar]<self.data['Lower_1'].iloc[bar]:
                self.go_long(bar, amount=min(self.current_balance/10,0))
            if self.position == -1 and self.data['Close'].iloc[bar]<self.data['Upper_2'].iloc[bar]:
                self.go_short(bar, amount='all')
            elif self.position == -1 and self.data['Close'].iloc[bar]<self.data['Upper_1'].iloc[bar]:
                self.go_long(bar, amount=min(self.current_balance/10,0))
            if self.trades != trade_flag:
                self.print_current_nav(bar)
        self.close_pos(bar+1) # close position at the last bar
        
    def plot_bol(self):
        plot_list = list(self.data.columns)
        plot_list.remove('returns')
        for cols in plot_list:
            self.data[cols].plot(figsize = (12, 8), title = self.symbol)
            plt.legend()
        
   
        
    def test_boll_strategy(self, SMA, dev):
        ''' 
        Backtests a Bollinger Bands mean-reversion strategy.
        
        Parameters
        ----------
        SMA: int
            moving window in bars (e.g. days) for simple moving average.
        dev: int
            distance for Lower/Upper Bands in Standard Deviation units
        '''
        
        # nice printout
        stm = "Testing Bollinger Bands Strategy | {} | SMA = {} & dev = {}".format(self.symbol, SMA, dev)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data() # reset dataset
        
        # prepare data
        self.data["SMA"] = self.data["Close"].rolling(SMA).mean()
        self.data["Lower"] = self.data["SMA"] - self.data["Close"].rolling(SMA).std() * dev
        self.data["Upper"] = self.data["SMA"] + self.data["Close"].rolling(SMA).std() * dev
        self.data.dropna(inplace = True) 
        
        # Bollinger strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            if self.position == 0: # when neutral
                if self.data["Close"].iloc[bar] < self.data["Lower"].iloc[bar]: # signal to go long
                    self.go_long(bar, amount = "all") # go long with full amount
                    self.position = 1  # long position
                elif self.data["Close"].iloc[bar] > self.data["Upper"].iloc[bar]: # signal to go Short
                    self.go_short(bar, amount = "all") # go short with full amount
                    self.position = -1 # short position
            elif self.position == 1: # when long
                if self.data["Close"].iloc[bar] > self.data["SMA"].iloc[bar]:
                    if self.data["Close"].iloc[bar] > self.data["Upper"].iloc[bar]: # signal to go short
                        self.go_short(bar, amount = "all") # go short with full amount
                        self.position = -1 # short position
                    else:
                        self.sell_instrument(bar, units = self.units) # go neutral
                        self.position = 0
            elif self.position == -1: # when short
                if self.data["Close"].iloc[bar] < self.data["SMA"].iloc[bar]:
                    if self.data["Close"].iloc[bar] < self.data["Lower"].iloc[bar]: # signal to go long
                        self.go_long(bar, amount = "all") # go long with full amount
                        self.position = 1 # long position
                    else:
                        self.buy_instrument(bar, units = -self.units) # go neutral
                        self.position = 0                
        self.close_pos(bar+1) # close position at the last bar