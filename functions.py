import pandas as pd
import numpy as np
import os
import typing

from typing import List, Dict, Any, Optional
from tabulate import tabulate
from talib import RSI
from num2words import num2words
from pandas import DataFrame


def read_to_df(ticker: str, tf: str, update: bool =False) -> DataFrame:
    """Function to read the data in from pickle or json depending on 'update'."""

    cur_path = os.path.dirname(__file__)

    if update == True:
        try:
            os.remove(f'Data/pickleddata/mex_{ticker.upper()}-{tf}')
        except:
            pass

    try:
        df = pd.read_pickle(f'Data/pickleddata/mex_{ticker.upper()}-{tf}')
    except:
        if tf == '1d':
            print('trying resampling')
            read_tf = '1h'
            try:
                if update == True:
                    os.remove(f'Data/pickleddata/mex_{ticker.upper()}-{read_tf}')

            except:
                print('no file to delete')
            df = pd.read_json(f'Data/jsons/{ticker.upper()}-{read_tf}.json')

            # key→old name, value→new name
            col_dict = {0: 'date', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'trades', 6: 'volume',7: 'vwap'}
            df.columns = [col_dict.get(x, x) for x in df.columns]

            # add ten hours for australian time
            df['date'] = pd.to_datetime(df['date'], unit='s')
            df['date'] = df['date'] + pd.Timedelta(hours=10)

            df = df.set_index('date', drop=True)
            dfcopy = df.copy()

            # dfcopy['date'] = df['date'].resample('1D').first()
            dfcopy['open'] = df['open'].resample('1D').first()
            dfcopy['high'] = df['high'].resample('1D').max()
            dfcopy['low'] = df['low'].resample('1D').min()
            dfcopy['close'] = df['close'].resample('1D').last()
            dfcopy['trades'] = df['trades'].resample('1D').sum()
            dfcopy['volume'] = df['volume'].resample('1D').sum()
            dfcopy['vwap'] = df['vwap'].resample('1D').mean()

            dfcopy = dfcopy.reset_index(drop=False)
            dfcopy = dfcopy.dropna()
            df = dfcopy.reset_index(drop=True)

        else:
            df = pd.read_json(f'Data/jsons/{ticker.upper()}-{tf}.json')

            # add column value names
            col_dict = {0: 'date', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'trades', 6: 'volume', 7: 'vwap'}  # #  key→old name, value→new name
            df.columns = [col_dict.get(x, x) for x in df.columns]


            # add 10 hours to change australian time to UTC.
            df['date'] = pd.to_datetime(df['date'], unit='s')
            df['date'] = df['date'] + pd.Timedelta(hours=10)

        # add colour of candle
        red_index = df.index[df['open'] > df['close']].to_list()
        green_index = df.index[df['open'] < df['close']].to_list()

        df['colour'] = 0
        x = df
        df.loc[green_index, 'colour'] = 'gr'
        df.loc[red_index, 'colour'] = 'r'

        # df['date'] = datetime.fromtimestamp(df['date'])
        df.to_pickle(f'Data/pickleddata/mex_{ticker.upper()}-{tf}')
        # df.to_pickle(os.path.relpath(f'..\\Data\\pickleddata\\mex_{ticker.upper()}-{tf}', cur_path))
    return df


def load_htf_data(df: DataFrame) -> (DataFrame, DataFrame):
    """A function to compute generate higher timeframe data"""
    freq = '1d'
    mins = 1440
    second_df = df.copy(deep=True)
    second_df['join_time'] = second_df['date']
    second_df.set_index('join_time', inplace=True)
    second_df.index = second_df.index.floor(freq)
    second_df.index = second_df.index - pd.Timedelta(mins, unit='m')
    second_df.reset_index(inplace=True)

    df2 = df.copy()
    df2 = df2[['date', 'close']]
    df2 = df2.set_index('date', drop=True)

    # create the HTF dataframe
    df2['close'] = df2.resample('1D').last()
    df2 = df2.reset_index(drop=False)
    df2 = df2.dropna()

    return df2, second_df


def join_htf_data(df2: DataFrame, second_df: DataFrame) -> DataFrame:
    """A function to merge low and high timeframe dataframes together"""
    # Force suffixes
    df2.columns = df2.columns.map(lambda x: str(x) + '_' + '1d')
    # df2['slow'] = df2['slow_1d']
    # df2['fast'] = df2['fast_1d']

    # Merge HTF data with current timeframe
    dataframe = pd.merge_asof(second_df, df2,
                              left_on='join_time',
                              right_on='date_' + '1d',
                              direction='backward',
                              suffixes=('', '')
                              )

    # trim DataFrame again
    dataframe.drop('join_time', inplace=True, axis=1)
    dataframe.drop('date_1d', inplace=True, axis=1)
    dataframe.reset_index(drop=False)

    return dataframe


def ret_sampled_fracs(*args, df: DataFrame, df_tot: DataFrame, df_occ: DataFrame, title: str, description: str) -> None:
    """A function to highlight the distribution of occurences of signals within a year, i.e. quarter, half. """

    df = df.set_index('date', drop=False)
    df_tot = df_tot.set_index('date', drop='False')
    df_occ = df_occ.set_index('date', drop='False')

    month_date = df.index.max() - pd.Timedelta(days=30)
    quarter_date = df.index.max() - pd.Timedelta(days=91)
    half_date = df.index.max() - pd.Timedelta(days=182)
    year_date = df.index.max() - pd.Timedelta(days=365)

    month_tot = df_tot[month_date : df.index.max()].shape[0]
    quarter_tot = df_tot[quarter_date : df.index.max()].shape[0]
    half_tot = df_tot[half_date : df.index.max()].shape[0]
    year_tot = df_tot[year_date : df.index.max()].shape[0]
    tot = df_tot.shape[0]

    month_occ = df_occ[month_date : df.index.max()].shape[0]
    quarter_occ = df_occ[quarter_date : df.index.max()].shape[0]
    half_occ = df_occ[half_date : df.index.max()].shape[0]
    year_occ = df_occ[year_date : df.index.max()].shape[0]
    occ = df_occ.shape[0]

    try:
        month_frac = month_occ/month_tot
    except:
        month_frac = 0

    try:
        quarter_frac = quarter_occ/quarter_tot
    except:
        quarter_frac = 0


    half_frac = half_occ/half_tot
    year_frac = year_occ/year_tot
    tot_frac = occ/tot

    print('--------'+str(title)+'---------')

    print(str(description))

    table = [["Month", month_frac, month_tot],["Quarter", quarter_frac, quarter_tot], ["Half", half_frac, half_tot], ["Year", year_frac, year_tot], ["Total", tot_frac, tot]]
    print(tabulate(table, headers=["Sample Window", "Fraction of Success", "Number of Occurences"]))


def add_opens_days(df: DataFrame) -> DataFrame:
    """A function to add the daily opens, weekly opens etc to the main df"""

    # add days to the dataframe, to match the dates
    df['day'] = df['date'].dt.day_name()

    # create a new column called opens to house
    df['opens'] = 0


    df2 = df.copy()
    df2 = df2[['date']]
    df2['index'] = df2.index.values
    df2 = df2.set_index('date', drop=True)

    # find indexes for 00:00 open/close
    indexer_open = df2.at_time('00:00')
    index_open = indexer_open.set_index('index').index

    # find index for london open
    indexer_ldn_open = df2.at_time('09:00')
    index_ldn_open = indexer_ldn_open.set_index('index').index

    # find index for london close
    indexer_ldn_close = df2.at_time('16:00')
    index_ldn_close = indexer_ldn_close.set_index('index').index

    # use indexes to values to dataframe
    df.loc[index_open, 'opens'] = 'open'
    df.loc[index_ldn_open, 'opens'] = 'ldn_open'
    df.loc[index_ldn_close, 'opens'] = 'ldn_close'

    return df

def apply_WO(df: DataFrame) -> DataFrame:
    """A function to add weekly opens to the main df"""

    def find_WO(row):

        if ((row['day'] == 'Monday') & (row['opens'] == 'open')):
            return True
        return False
    df['weekly_open'] = df.apply(find_WO, axis=1)
    return df

def apply_MO(df: DataFrame) -> DataFrame:
    """A function to add monthly opens to the main df"""

    df['monthly_end'] = False
    df.loc[(df['date'].dt.is_month_end), 'monthly_end'] = True
    df['monthly_end'] = df['monthly_end'].shift(1)

    df['monthly_start'] = 0
    df.loc[(df['date'].dt.is_month_start), 'monthly_start'] = True

    df['monthly_open'] = False
    df.loc[((df['monthly_start'] == True) & (df['monthly_end'] == True)), 'monthly_open'] = True

    df = df.drop('monthly_start', axis=1)
    df = df.drop('monthly_end', axis=1)

    return df

def calc_range(df: DataFrame) -> DataFrame:
    """A function to calculate ranges of candles"""
    df['range'] = df['high'] - df['low']
    return df

def weekday_adjusted_vol_range(df: DataFrame) -> DataFrame:
    """A function to calculate volume ranges, adjusted for days of the week"""
    df['weekday_vol'] = 0
    df['weekday_range'] = 0

    # set index to date so we can group by days
    df = df.set_index('date', drop=False)

    # group by days and use transform of a rolling mean
    df['weekday_vol'] = df['volume'].groupby(df.index.weekday).transform(lambda x: x.rolling(window=5).mean())

    # group by days and use transform of a rolling mean
    df['weekday_range'] = df['range'].groupby(df.index.weekday).transform(lambda x: x.rolling(window=5).mean())

    # revert back to normal index
    df = df.reset_index(drop=True)
    return df



def vol_range(df: DataFrame) -> DataFrame:
    """A function to calculate volume and range over a 30 day rolling period"""

    # calculate rolling 30 day average of range and volatility
    df['avg_range'] = df['range'].rolling(window=30).mean()
    df['avg_vol'] = df['volume'].rolling(window=30).mean()

    return df

def calc_percentile(df: DataFrame) -> DataFrame:
    """A function to calculate percentiles of values, specifically volume and range"""

    df['rel_vol'] = df['volume']/df['avg_vol']
    df['percentile_vol'] = df['rel_vol'].rank(pct=True)

    df['rel_range'] = df['range']/df['avg_range']
    df['percentile_range'] = df['rel_range'].rank(pct=True)

    # set index to date so we can group by days
    df = df.set_index('date', drop=False)

    # calculate percentiles from only that day
    df['rel_weekday_vol'] = df['volume']/df['weekday_vol']
    df['percentile_weekday_vol'] = df['rel_weekday_vol'].groupby(df.index.weekday).transform(lambda x: x.rank(pct=True))

    # calculate percentiles from only that day
    df['rel_weekday_range'] = df['range']/df['weekday_range']
    df['percentile_weekday_range'] = df['rel_weekday_range'].groupby(df.index.weekday).transform(lambda x: x.rank(pct=True))

    # revert back to normal index
    df = df.reset_index(drop=True)

    return df


def calc_vwap(df: DataFrame) -> DataFrame:
    """A function to calculate volume weighted average price for weeks and months, anchored at the opens"""

    df = df[['date', 'close', 'volume', 'vwap', 'weekly_open', 'monthly_open']]
    df2 = df.copy()
    df_weekly_indices = df[df['weekly_open'] == True].index
    df_monthly_indices = df[df['monthly_open'] == True].index
    df['orig_index'] = df.index

    weekly_df_list = []

    cutoff_end = df.loc[df_weekly_indices[0], 'date']
    cutoff_start = df.loc[0, 'date']
    x = cutoff_start
    y = cutoff_end
    df2 = df.set_index('date', drop=False)
    weekly_df_cutoff = df2[str(cutoff_start):str(cutoff_end)]
    weekly_df_cutoff = weekly_df_cutoff.reset_index(drop=True)
    weekly_df_cutoff = weekly_df_cutoff.drop(columns='orig_index')
    weekly_df_list.append(weekly_df_cutoff)

    for weekly_index in df_weekly_indices:

        start_date = df.loc[weekly_index, 'date']
        end_date = df.loc[weekly_index, 'date'] + pd.Timedelta(days=7)

        subset_df = df2[str(start_date):str(end_date)]
        week_window = subset_df.shape[0] - 1

        subset_df['Wvwap'] = (((subset_df['vwap'] * subset_df['volume']).rolling(week_window, min_periods=1)).sum())/(subset_df['volume'].rolling(week_window, min_periods=1).sum())


        peep = subset_df.set_index('orig_index', drop=True)

        weekly_df_list.append(peep)


    df3 = pd.concat(weekly_df_list, sort=False)
    df3 = df3.loc[~df3.index.duplicated(keep='last')]
    df_weekly_series = df3['Wvwap']


    cutoff_end_mon = df.loc[df_monthly_indices[0], 'date']
    cutoff_start_mon = df.loc[0, 'date']
    monthly_df_list = []

    monthly_df_cutoff = df2[str(cutoff_start_mon):str(cutoff_end_mon)]
    monthly_df_cutoff = monthly_df_cutoff.reset_index(drop=True)
    monthly_df_cutoff = monthly_df_cutoff.drop(columns='orig_index')
    monthly_df_list.append(monthly_df_cutoff)

    i = 0
    for monthly_index in df_monthly_indices:

        start_date = df.loc[monthly_index, 'date']
        try:
            end_date = df.loc[df_monthly_indices[i+1], 'date']
        except:
            end_date = df.loc[df.index.max(), 'date']

        subset_df_mon = df2[str(start_date):str(end_date)]

        mon_window = subset_df_mon.shape[0] - 1
        subset_df_mon['Mvwap'] = ((subset_df_mon['vwap'] * subset_df_mon['volume']).rolling(mon_window, min_periods=0).sum())\
                                 /(subset_df_mon['volume'].rolling(mon_window, min_periods=0).sum())
        # subset_df['Wvwap2'] = ((subset_df['close'] * subset_df['volume']).sum())/(subset_df['volume'].sum())
        subset_df_mon = subset_df_mon.set_index('orig_index', drop=False)
        monthly_df_list.append(subset_df_mon)

        i += 1

    df4 = pd.concat(monthly_df_list, sort=False)
    df4 = df4.loc[~df4.index.duplicated(keep='last')]
    df4 = df4.join(df_weekly_series)
    df4 = df4.drop(columns='orig_index')

    df4 = df4[['date', 'Wvwap', 'Mvwap']]
    df4 = df4.fillna(0)
    df4 = df4.set_index('date', drop=False)
    vwaps = pd.DataFrame()
    vwaps['W_vwap'] = df4['Wvwap'].resample('1D').last()
    vwaps['M_vwap'] = df4['Mvwap'].resample('1D').last()
    vwaps['date'] = df4['date'].resample('1D').last()

    vwaps = vwaps.reset_index(drop=True)

    # add the most recent values to bottom of vwaps so we have a current vwaps value rather than most recent close
    df4 = df4.reset_index(drop=True)

    Wvwap = df4['Wvwap']
    Mvwap = df4['Mvwap']
    date = df4['date']

    last_Wvwap= Wvwap.iloc[-1]
    last_Mvwap = Mvwap.iloc[-1]
    last_date = date.iloc[-1]

    list = pd.DataFrame({"W_vwap":[last_Wvwap],"M_vwap":[last_Mvwap], "date":[last_date]})

    vwaps = vwaps.append(list, ignore_index=True)

    return vwaps


def calc_ma(df: DataFrame) -> DataFrame:
    """A function to calculate moving averages"""
    df['5d_ma'] = df['close'].rolling(5).mean()
    df['10d_ma'] = df['close'].rolling(10).mean()
    df['20d_ma'] = df['close'].rolling(20).mean()
    df['50d_ma'] = df['close'].rolling(50).mean()
    df['100d_ma'] = df['close'].rolling(100).mean()

    return df


def calc_metrics(df: DataFrame, metric: str) -> List:
    """A function to output useful metrics of signals, i.e. sharpe ratio"""
    df2 = df.copy()
    no_wins = df2[df2['colour'] == 'gr'].shape[0]
    no_loss = df2[df2['colour'] == 'r'].shape[0]

    try:
        win_percent = (no_wins/(no_wins + no_loss)) * 100
    except:
        win_percent = 0

    mean_gain = df2['returns'][df2['returns'] > 0].mean() * 100
    mean_loss = df2['returns'][df2['returns'] < 0].mean() * 100

    risk_free = 0.00
    returns = df2['returns'].mean()
    stddev = df2['returns'].std()

    sharpe = ((returns - risk_free)/(stddev)) * np.sqrt(365)

    df2 = df2.set_index('date', drop=True)

    pass

    return [metric, no_wins + no_loss, no_wins, no_loss, win_percent, sharpe,  mean_gain, mean_loss]


def biases(df: DataFrame) -> (List, List, DataFrame):
    """A function to populate the main df with signals bool values"""

    df['rsi_2'] = RSI(df['close'], 2)
    df['rsi_14'] = RSI(df['close'], 14)

    signal_names = ['buysharpe', 'ma_50', 'ma_100', 'Mvwap', 'Wvwap', 'rsi_2', 'rsi_14']
    for signal in signal_names:
        df[f'signal_{str(signal)}'] = False

    # calc sharpe of buy and hold
    buysharpe = calc_metrics(df, 'buy&hold')
    df['signal_buysharpe'] = True

    # calc sharpe of close above vwap and show True
    df_Wvwap = df[df['close'].shift(1) > df['W_vwap'].shift(1)]
    df.loc[df['close'].shift(1) > df['W_vwap'].shift(1), 'signal_Wvwap'] = True
    Wvwap = calc_metrics(df_Wvwap, '>Wvwap')

    df_Mvwap = df[df['close'].shift(1) > df['M_vwap'].shift(1)]
    df.loc[df['close'].shift(1) > df['M_vwap'].shift(1), 'signal_Mvwap'] = True
    Mvwap = calc_metrics(df_Mvwap, '>Mvwap')

    # calc metrics of price closing above 50ma, 100ma and show True
    df_50ma = df[df['close'].shift(1) > df['50d_ma'].shift(1)]
    df.loc[df['close'].shift(1) > df['50d_ma'].shift(1), 'signal_ma_50'] = True
    ma_50 = calc_metrics(df_50ma, '>50d_ma')

    df_100ma = df[df['close'].shift(1) > df['100d_ma'].shift(1)]
    df.loc[df['close'].shift(1) > df['50d_ma'].shift(1), 'signal_ma_100'] = True
    ma_100 = calc_metrics(df_100ma, '>100d_ma')

    # calc metrics of rsi_2 > 90 and show True
    df_rsi_2 = df[df['rsi_2'].shift(1) > 90]
    df.loc[df['rsi_2'].shift(1) > 90, 'signal_rsi_2'] = True
    rsi_2 = calc_metrics(df_rsi_2, 'RSI_2>90')

    df_rsi_14 = df[df['rsi_14'].shift(1) > 70]
    df.loc[df['rsi_14'].shift(1) > 70, 'signal_rsi_14'] = True
    rsi_14 = calc_metrics(df_rsi_14, 'RSI_14>70')

    greens = df[df['colour'] == 'gr']
    reds = df[df['colour'] == 'r']

    # list of lists containing metrics of all active signals
    signals = [buysharpe, ma_50, ma_100, Mvwap, Wvwap, rsi_2, rsi_14]

    return signal_names, signals, df

def create_table(df: DataFrame) -> (tabulate, DataFrame, List, List):
    """A function to create a table with all active signals"""

    # get signals and their names from typical htf biases, i.e price above/below vwap/MA.
    signal_names, signals, df = biases(df)

    # get signals from statistics generated from looping variables
    metrics, df = get_signal_list(df)

    # print(len(signal_names_stat))
    # print(len(stats_table[2]))

    # for i in range(len(signal_names_stat)):
    #      df = df.rename(columns={str(signal_names_stat[i]): str(stats_table[0][i])})

    # last list within stats_table is the description/name
    #signal_names.extend(stats_table[3])

    # for each signal name, calculate the metric values and append the list to the current signal list.
    # for i in range(len(stats_table[0])):
    #     signals.append(calc_metrics(df[df['signal_' + str(stats_table[3][i])] == True], str(stats_table[0][i])))


    for metric in metrics:
        signals.append(metric)
        signal_names.append(metric[0])


    # check if signals are active and return new signal list
    signals = check_signals(df, signals, signal_names)

    signals_test = signals

    for i in range(len(signals)):
        signals[i] = signals[i][:-1]

    for signal in signals:
        for i in range(1, len(signal)):
            signal[i] = np.format_float_positional(signal[i], precision=4, unique=False, fractional=False, trim='k')


    # sort signals based on number of occurences
    # create list of occurences
    occ = []
    for signal in signals:
        occ.append(signal[1])
    # zip two lists together, sort based on occurences
    table = [x for _, x in sorted(zip(occ, signals), reverse=False)]



    # table = [buysharpe, ma_50, ma_100, Mvwap, Wvwap, rsi_2, rsi_14]
    print(tabulate(table, headers=["Active", "#", "#W", "#L", "Win%", "Sharpe", "Av Win %",
                                   "Av Loss %"]))

    columns = ["Active", "#", "#W", "#L", "Win%", "Sharpe", "Av Win %", "Av Loss %"]
    # table.insert(0, ["Active Signals", "# Occurences", "#Wins", "#Loss", "Win%", "Sharpe", "Average Win %", "Average Loss %"])

    return table, df, signals_test, columns

def check_signals(df: DataFrame, signals: List, signal_names: List) -> List:
    """A function to check which signals are active"""

    x = len(signal_names)
    y = len(signals)
    new_list = []
    for i in range(len(signal_names)):
        # .loc[-1] is the current day, want prior daily close

        if df[f'signal_{str(signal_names[i])}'].iloc[-2]:
            print(f'signal_{str(signal_names[i])} is active --- {str(signals[i])}')

            # add signal name to list, then add list to filtered list
            signals[i].append(signal_names[i])
            new_list.append(signals[i])

    return new_list

def find_percentiles(df: DataFrame) -> DataFrame:
    """A function that runs all functions required for percentile values"""
    df = add_opens_days(df)
    df = calc_range(df)
    df = weekday_adjusted_vol_range(df)
    df = vol_range(df)
    df = calc_percentile(df)

    return df

def find_returns(df: DataFrame) -> DataFrame:
    """A function that calculates returns from one day (inverse contracts)"""
    df['returns'] = df['close'] * ((1/df['open']) - (1/df['close']))

    return df

def find_gradient(df: DataFrame) -> DataFrame:
    """A function to calculate gradients of moving averages"""
    df['grad_5d_ma'] = df['5d_ma'] - df['5d_ma'].shift()
    df['grad_10d_ma'] = df['10d_ma'] - df['10d_ma'].shift()
    df['grad_20d_ma'] = df['20d_ma'] - df['20d_ma'].shift()

    return df

def find_body_ratio(df: DataFrame) -> DataFrame:
    """A function to calculate body to wick ratio"""
    # if green find length of upper wick and vice versa
    df['ratio'] = 0

    df['ratio'].loc[df['close'] > df['open']] = abs((df['close'] - df['open'])/(df['high'] - df['close']))

    df['ratio'].loc[df['close'] < df['open']] = abs((df['close'] - df['open'])/(df['low'] - df['close']))

    return df

def find_volume_ratio(df: DataFrame) -> DataFrame:
    """A function to find the ratio of current volume with previous candle volume"""
    df['volume_ratio'] = df['volume']/(df['volume'].shift(1))

    return df


def get_signal_list(df: DataFrame) -> (List, tabulate, DataFrame, List):
    """"""

    columns = create_truth_columns(df)
    signals_2 = create_success_list(columns)

    metric_list = []

    for signal in signals_2:
        dfcopy = df.loc[signal[0]]

        # Isolate metrics and compute one string
        metrics = signal[3:]
        metric_name = ''
        for value in metrics:
            metric_name += str(value) + '+'

        if metric_name.endswith('+'):
            metric_name = metric_name[:-1]

        df['signal_' + metric_name] = False
        df['signal_' + metric_name].loc[signal[0]] = True

        metric = calc_metrics(dfcopy, metric_name)
        metric_list.append(metric)


    for val in metric_list:
        if val[5] < 0:
            metric_list.remove(val)

    return metric_list, df


def create_truth_columns(df):
    """Function to populate df with signal bool values prior to the assignment of their actual bool values"""

    # The signal checker, when seeing if values are True,

    # signal 1, price above or below dma

    df['^_5dma'] = False
    df.loc[df['close'] > df['5d_ma'], '^_5dma'] = True

    df['^_10dma'] = False
    df.loc[df['close'] > df['10d_ma'], '^_10dma'] = True

    df['^_20dma'] = False
    df.loc[df['close'] > df['20d_ma'], '^_20dma'] = True

    df['b_5dma'] = False
    df.loc[df['close'] < df['5d_ma'], 'b_5dma'] = True

    df['b_10dma'] = False
    df.loc[df['close'] < df['10d_ma'], 'b_10dma'] = True

    df['b_20dma'] = False
    df.loc[df['close'] < df['20d_ma'], 'b_20dma'] = True

    df['ignore_dma'] = True

    # signal 2, gradient sloping up or down
    df['^_g5'] = False
    df.loc[df['grad_5d_ma'] > 0, '^_g5'] = True

    df['^_g10'] = False
    df.loc[df['grad_10d_ma'] > 0, '^_g10'] = True

    df['^_g20'] = False
    df.loc[df['grad_20d_ma'] > 0, '^_g20'] = True

    df['d_g5'] = False
    df.loc[df['grad_5d_ma'] < 0, 'd_g5'] = True

    df['d_g10'] = False
    df.loc[df['grad_10d_ma'] < 0, 'd_g10'] = True

    df['d_g20'] = False
    df.loc[df['grad_20d_ma'] < 0, 'd_g20'] = True

    df['ignore_grad'] = True

    # signal 3 prev candle colour
    df['p_r'] = False
    df.loc[(df['colour'] == 'r'), 'p_r'] = True

    df['p_2_r'] = False
    df.loc[(df['colour'] == 'r') & (df['colour'].shift(1) == 'r'), 'p_2_r'] = True

    df['p_3_r'] = False
    df.loc[(df['colour'] == 'r') & (df['colour'].shift(1) == 'r') & (df['colour'].shift(2) == 'r'), 'p_3_r'] = True

    df['p_gr'] = False
    df.loc[(df['colour']== 'gr'), 'p_gr'] = True

    df['p_2_gr'] = False
    df.loc[(df['colour'] == 'gr') & (df['colour'].shift(1) == 'gr'), 'p_2_gr'] = True

    df['p_3_gr'] = False
    df.loc[(df['colour'] == 'gr') & (df['colour'].shift(1) == 'gr') & (df['colour'].shift(2) == 'gr'), 'p_3_gr'] = True

    df['ignore_prev'] = True

    # signal 4 candle:wick ratio
    df['cndl_^_125'] = False
    df['cndl_b_075'] = False

    # look to previous candle
    df.loc[df['ratio'] > 1.25, 'cndl_^_125'] = True
    df.loc[df['ratio'] < 0.75, 'cndl_b_075'] = True

    df['ignore_candle'] = True

    # signal 5, volume:prev_volume ratio
    df['vol_^_125'] = False
    df['vol_b_075'] = False

    df.loc[df['volume_ratio'] > 1.25, 'vol_^_125'] = True
    df.loc[df['volume_ratio'] < 0.75, 'vol_b_075'] = True

    df['ignore_volume'] = True

    return df


def create_success_list(df: DataFrame) -> List:
    """A function to loop through all the stat conditions against each other to try find signals with high success rate"""

    # lists of conditions that need to be tested against each other, true/false values applied in above function
    dma_list = ['^_5dma', '^_10dma', 'b_5dma', 'b_10dma', 'ignore_dma']
    grad_list = ['^_g5', '^_g10', 'd_g5', 'd_g10','ignore_grad']
    prev = ['p_r', 'p_2_r', 'p_3_r', 'p_gr', 'p_2_gr', 'p_3_gr', 'ignore_prev']
    # candle_ratio = ['candle_above_125', 'candle_below_075', 'ignore_candle']
    candle_ratio = ['ignore_candle']
    volume_ratio = ['vol_^_125', 'vol_b_075', 'ignore_volume']
    # volume_ratio = ['ignore_volume']
    daily_returns = ['']
    outcome = ['gr', 'r']

    signal_list = [dma_list, grad_list, prev]

    successes = []

    i = 0
    for signala in dma_list:
        for signalb in grad_list:
            for signalc in prev:
                for signald in candle_ratio:
                    for signale in volume_ratio:
                        for signalf in outcome:

                            # total number of occurences that these conditions all align on previous candle
                            tot = df[(df[str(signala)].shift(1) == True) & (df[str(signalb)].shift(1) == True) & (df[str(signalc)].shift(1) == True)
                                      & (df[str(signald)].shift(1) == True) & (df[str(signale)].shift(1) == True)].shape[0]

                            # total number of occurences that these conditions all
                            # align on previous candle and the price goes one direction in next candle
                            occ = df[(df[str(signala)].shift(1) == True) & (df[str(signalb)].shift(1) == True) & (df[str(signalc)].shift(1) == True)
                                      & (df[str(signald)].shift(1) == True) & (df[str(signale)].shift(1) == True) & (df['colour'] == signalf)].shape[0]

                            # index of positions to pass through to output function
                            truth_index = df[(df[str(signala)].shift(1) == True) & (df[str(signalb)].shift(1) == True) & (df[str(signalc)].shift(1) == True)
                                      & (df[str(signald)].shift(1) == True) & (df[str(signale)].shift(1) == True)].index

                            try:
                                frac = occ/tot
                            except:
                                frac = 0

                            if (((frac > 0.55) & (tot > 100)) or ((frac > 0.6) & (tot > 80)) or ((frac > 0.65) & (tot > 60))):
                                list = []
                                list.append(truth_index)
                                list.append(frac)
                                list.append(tot)

                                if signala is not dma_list[-1]:
                                    list.append(signala)
                                if signalb is not grad_list[-1]:
                                    list.append(signalb)
                                if signalc is not prev[-1]:
                                    list.append(signalc)
                                if signald is not candle_ratio[-1]:
                                    list.append(signald)
                                if signale is not volume_ratio[-1]:
                                    list.append(signale)

                                list.append(signalf)

                                successes.append(list)

    return successes

