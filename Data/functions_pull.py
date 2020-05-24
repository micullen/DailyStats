import arrow
import logging
import json
import datetime
import os
from typing import List, NamedTuple, Optional, Tuple, Any, Dict
from arguments import TimeRange
import requests
from datetime import datetime
import logging
import time
import os

logging.basicConfig(level=logging.DEBUG)


TICKER_INTERVAL_MINUTES = {
    '1m': 1,
    '3m': 3,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '2h': 120,
    '3h': 180,
    '4h': 240,
    '6h': 360,
    '8h': 480,
    '12h': 720,
    '1d': 1440,
    '3d': 4320,
    '1w': 10080,
}





def download_backtesting_testdata(datadir: str,
                                  pair: str,
                                  tick_interval: str = '5m',
                                  timerange: Optional[TimeRange] = None) -> None:

    """
    Download the latest ticker intervals from the exchange for the pairs passed in parameters
    The data is downloaded starting from the last correct ticker interval data that
    esists in a cache. If timerange starts earlier than the data in the cache,
    the full data will be redownloaded

    Based on @Rybolov work: https://github.com/rybolov/freqtrade-data
    :param pairs: list of pairs to download
    :param tick_interval: ticker interval
    :param timerange: range of time to download
    :return: None

    """

    path = make_testdata_path(datadir)
    filepair = pair.replace("/", "_")
    filename = os.path.join(path, f'{filepair}-{tick_interval}.json')

    data, since_s = load_cached_data_for_updating(filename, tick_interval, timerange)

    new_data = get_candle_history(pair=pair, tick_interval=tick_interval, since_s=since_s)
    data.extend(new_data)

    file_dump_json(filename, data)


def file_dump_json(filename, data, is_zip=False) -> None:
    """
    Dump JSON data into a file
    :param filename: file to create
    :param data: JSON Data to save
    :return:
    """
    print(f'dumping json to "{filename}"')

    if is_zip:
        if not filename.endswith('.gz'):
            filename = filename + '.gz'
        with gzip.open(filename, 'w') as fp:
            json.dump(data, fp, default=str)
    else:
        with open(filename, 'w') as fp:
            json.dump(data, fp, default=str)


def format_ms_time(date: int) -> str:
    """
    convert MS date to readable format.
    : epoch-string in ms
    """
    return datetime.fromtimestamp(date/1000.0).strftime('%Y-%m-%dT%H:%M:%S')

def load_cached_data_for_updating(filename: str,
                                  tick_interval: str,
                                  timerange: Optional[TimeRange]) -> Tuple[
                                                                                  List[Any],
                                                                                  Optional[int]]:
    """
    Load cached data and choose what part of the data should be updated
    """

    since_s = None
    print(f'filename is {filename}')

    # user sets timerange, so find the start time
    if timerange:
        if timerange.starttype == 'date':
            start_date = datetime.fromtimestamp(timerange.startts)
            logging.debug(f'start_date in s is {start_date}')

            since_s = timerange.startts

        elif timerange.stoptype == 'line':
            num_minutes = timerange.stopts * TICKER_INTERVAL_MINUTES[tick_interval]
            since_s = arrow.utcnow().shift(minutes=num_minutes).timestamp

    # read the cached file
    if os.path.isfile(filename):
        with open(filename, "rt") as file:
            data = json.load(file)
            # remove the last item, because we are not sure if it is correct
            # it could be fetched when the candle was incompleted
            if data:
                data.pop()
    else:
        data = []

    if data:
        if since_s and since_s < data[0][0]:
            #print('data is [] ' + '*'*30)
            # the data is requested for earlier period than the cache has
            # so fully redownload all the data
            data = []
        else:
            #print('add 1000 to since_s' * 4)
            # a part of the data was already downloaded, so
            # download unexist data only
            since_s = data[-1][0] + 1
            pass

            #since_s = datetime.timestamp((data[-1][0])) + 1

    return data, since_s



def get_candle_history(pair: str, tick_interval: str,
                       since_s: Optional[int] = None, num_candles: int = 750) -> List[Dict]:

    # TODO: Add to config
    backoff_start = 4
    backoff_max = 6
    attempt = 0
    since_s = since_s

    # last item should be in the time interval [now - tick_interval, now]
    till_time_ms = arrow.utcnow().shift(
                    minutes=-TICKER_INTERVAL_MINUTES[tick_interval]
                ).timestamp
    # it looks as if some exchanges return cached data
    # and they update it one in several minute, so 10 mins interval
    # is necessary to keep downloading of an empty array when all
    # cached data was already downloaded

    # TODO: Not convinced we need this for Bitfinex
    #till_time_ms = min(till_time_ms, arrow.utcnow().shift(minutes=-10).timestamp * 1000)

    # TODO: If both since_ms and num_candles are set explicitly, num_candles will be ignored, make this clear

    # Bitfinex API v1 returns from the beginning of time if since_ms is not specified
    if not since_s:

        print(':0'*30)
        since_s = arrow.utcnow().shift(
            minutes=-(self._api.parse_timeframe(tick_interval) / 60) * num_candles
        ).timestamp

    # TODO: Declare this somewhere else, also 1000 is Bitfinex specific
    if num_candles > 750:
        lim = 750
    else:
        lim = num_candles

    data: List[Dict[Any, Any]] = []
    print(f'since_s')
    while not since_s or since_s < till_time_ms:
        try:
            print(f'since_s is {since_s}')
            start = datetime.fromtimestamp(since_s)
            print(f'start is {start}')
            data_part1 = requests.get(f'https://www.bitmex.com/api/v1/trade/bucketed?binSize={tick_interval}&symbol={pair}&count={num_candles}&startTime={start}')
            data_part = data_part1.json()
            print(data_part1)
            for index in range(len(data_part)):
                dicty = data_part[index]
                #print(f'dicty is {dicty}')
                data_list = list(dicty.values())
                new_dt = data_list[0]
                new_dt = new_dt[:-5]
                new_dt = new_dt[:10] + '' + new_dt[11:]
                new_dt_str = datetime.strptime(new_dt, '%Y-%m-%d%H:%M:%S')
                new_ts = datetime.timestamp(new_dt_str)
                # = [time, open, high, low, close, trades, volume, vwap]
                # = [0,     2  ,   3 ,  4 ,  5   ,   6   ,    7,    8  ]
                tohlcv_data_list = [new_ts, data_list[2], data_list[3], data_list[4], data_list[5], data_list[6], data_list[7], data_list[8]]
                data_part[index] = tohlcv_data_list

        except Exception as e:
            b = backoff_start + attempt
            if b > backoff_max:
                b = backoff_max
            #t = 10
            t = 2 ** b
            attempt += 1
            print('Paginator backing off for {} seconds'.format(t))
            time.sleep(t)
            continue

        # Because some exchange sort Tickers ASC and other DESC.
        # Ex: Bittrex returns a list of tickers ASC (oldest first, newest last)
        # when GDAX returns a list of tickers DESC (newest first, oldest last)
        data_part = sorted(data_part, key=lambda x: x[0])

        if not data_part:
            break

        data.extend(data_part)
        last_s = data[-1][0]
        since_s = last_s + 1
        #logger.info('Paginator progress: %d', len(data))
        print('Paginator progress: {}'.format(len(data)))

    return data



def make_testdata_path(datadir: str) -> str:
    """Return the path where testdata files are stored"""
    return datadir or os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '..', 'tests', 'testdata'
        )
    )