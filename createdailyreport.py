from functions import load_htf_data, join_htf_data, read_to_df, ret_sampled_fracs, apply_WO, apply_MO, add_opens_days,\
    calc_range, weekday_adjusted_vol_range, vol_range, calc_percentile, create_table, find_percentiles, find_body_ratio, find_volume_ratio, \
    calc_ma, find_gradient, calc_vwap, find_returns, get_signal_list, create_success_list, create_truth_columns

#from createdailystats2 import get_signal_list
from config import *
import telegram
import telebot
import datetime
from plotly import graph_objects as go
import plotly.figure_factory as ff
import os
import dash_table
from pandas import DataFrame
#import new data
#os.system(' python Data/pulldata.py')

def btc_sort():
    btc = read_to_df('xbtusd', '1d', update=True)
    btc = add_opens_days(btc)
    btc = apply_WO(btc)
    btc = apply_MO(btc)
    btc = find_body_ratio(btc)
    btc = find_volume_ratio(btc)
    btc = calc_ma(btc)
    btc = find_gradient(btc)

    btc_1m = read_to_df('xbtusd', '5m', update=True)
    btc_1m = add_opens_days(btc_1m)
    btc_1m = apply_WO(btc_1m)
    btc_1m = apply_MO(btc_1m)
    vwaps = calc_vwap(btc_1m)

    btc = btc.assign(W_vwap=vwaps[['W_vwap']])
    btc = btc.assign(M_vwap=vwaps[['M_vwap']])
    btc = find_returns(btc)

    table_btc, btc, signals_test, columns = create_table(btc)
    table_btc.insert(0, columns)
    btc = find_percentiles(btc)
    btc = find_body_ratio(btc)
    btc = find_volume_ratio(btc)
    btc = calc_ma(btc)
    btc = find_gradient(btc)

    fig_btc = dash_table.DataTable
    fig_btc = ff.create_table(table_btc)

    return btc, fig_btc


def save_report(df: DataFrame, fig, ticker: str) -> None:

    fig.add_trace(go.Indicator(
        domain = {'x': [0, 0.4], 'y': [0.8, 0.9]},
        value = df["percentile_vol"].iloc[-2]*100,
        number = {'suffix': "%"},
        mode = "gauge+number+delta",
        title = {'text': f"Relative Volume: {df['rel_vol'].iloc[-2]:.2f}"},
        delta = {'reference': df['percentile_vol'].iloc[-3]*100},
        gauge = {'axis': {'range': [None, 100]}}))

    fig.add_trace(go.Indicator(
        domain = {'x': [0, 0.4], 'y': [0.6, 0.7]},
        value = df["percentile_weekday_vol"].iloc[-2]*100,
        number = {'suffix': "%"},
        mode = "gauge+number+delta",
        title = {'text': f"Relative WkDay Volume: {df['rel_weekday_vol'].iloc[-2]:.2f}"},
        delta = {'reference': df['percentile_weekday_vol'].iloc[-3]*100},
        gauge = {'axis': {'range': [None, 100]}}))

    fig.add_trace(go.Indicator(
        domain = {'x': [0.5, 0.9], 'y': [0.8, 0.9]},
        value = df["percentile_range"].iloc[-2]*100,
        number = {'suffix': "%"},
        mode = "gauge+number+delta",
        title = {'text': f"Relative Range: {df['rel_range'].iloc[-2]:.2f}"},
        delta = {'reference': df['percentile_range'].iloc[-3]*100},
        gauge = {'axis': {'range': [None, 100]}}))

    fig.add_trace(go.Indicator(
        domain = {'x': [0.5, 0.9], 'y': [0.6, 0.7]},
        value = df["percentile_weekday_range"].iloc[-2]*100,
        number = {'suffix': "%"},
        mode = "gauge+number+delta",
        title = {'text': f"Relative WkDay Range: {df['rel_weekday_range'].iloc[-2]:.2f}"},
        delta = {'reference': df['percentile_weekday_range'].iloc[-3]*100},
        gauge = {'axis': {'range': [None, 100]}}))

    #fig_vol.show()


    # initialize xaxis2 and yaxis2
    fig['layout']['xaxis2'] = {}
    fig['layout']['yaxis2'] = {}


    # Edit layout for subplots
    fig.layout.yaxis.update({'domain': [0, 0.5]})
    fig.layout.yaxis2.update({'domain': [0.55, 0.9]})

    # The graph's yaxis2 MUST BE anchored to the graph's xaxis2 and vice versa
    fig.layout.yaxis2.update({'anchor': 'x2'})
    fig.layout.xaxis2.update({'anchor': 'y2'})
    fig.layout.xaxis.update({'title': 'Active Biases'})
    fig.layout.xaxis2.update({'title': 'Relative Volume and Ranges to Predict Market Conditions'})

    # Update the margins to add a title and see graph x-labels.
    #fig.layout.margin.update({'t':75, 'l':50})


    # Update the height because adding a graph vertically will interact with
    # the plot height calculated for the table
    fig.layout.update({'height':800})

    # Plot!
    fig.show()
    d = datetime.datetime.today()
    d = d.strftime('%d-%m-%Y')
    #plotly.offline.plot(fig, filename=f'daily_reports/{ticker}_{d}.html', auto_open=False)
    fig.write_image(f'daily_reports/{ticker}_{d}.png')



btc, fig_btc = btc_sort()

save_report(btc, fig_btc, 'xbt')

x = telebot
y = x
CHAT_ID = "-356386714"

bot = telegram.Bot(token="891232463:AAGkxYE9Bbwk2ityTI-TgcV0ghJ78V6xciY")

def send_message():
    """A function to send the telegram message containging"""
    d = datetime.datetime.today()
    d = d.strftime('%d-%m-%Y')
    #bot = telegram.Bot(token=TOKEN)
    #bot.sendDocument(CHAT_ID, document=f'Analyse/daily_reports/xbtusd_{d}.html')
    #bot.sendDocument(CHAT_ID, f'Analyse/daily_reports/xbtusd_{d}.html')
    bot.send_photo(chat_id=CHAT_ID, photo=open(f'daily_reports/xbt_{d}.png', 'rb'), caption=f"XBT Daily stats for {d}")
    #bot.send_document()

# Send
send_message()