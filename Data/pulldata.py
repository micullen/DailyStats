
import arrow
from functions_pull import *
import arguments
from arguments import TimeRange
import sys

"""Insert download path here, i.e. /Users/xyz/Trading/Stats"""
dl_path = '/Users/mc/Documents/DailyReports/Reports/Data/jsons/'

arguments = arguments.Arguments(sys.argv[1:], 'download utility')
arguments.testdata_dl_options()
args = arguments.parse_args()


"""variables, enter how many days you want to run over. for timeframes, i think the bitmex only gives out bucketed trades of
5m and 1d. but resampling using pandas is pretty easy. timeframes below uses 5m and 1d so will download both with same run if you like"""

# Order of results from client
reverse = True

#how many days of data you want
days = 1200

# What pairs and timeframes you wish to download
PAIRS = ["XBTUSD"]
timeframes = ['5m', '1h', '1d']

timerange = TimeRange()
time_since = arrow.utcnow().shift(days=-days)
time_since = time_since.strftime("%Y%m%d")
timerange = arguments.parse_timerange(f'{time_since}-')
print(f'timerange is {timerange}' + '-' * 20)



for pair in PAIRS:


    for tick_interval in timeframes:
        pair_print = pair.replace('/', '_')
        filename = f'{pair_print}-{tick_interval}.json'
        dl_file = dl_path + '/' + filename
        print(f'downloading pair {pair}, interval {tick_interval}')
        print(type(dl_file))
        download_backtesting_testdata(str(dl_path),
                                      pair=pair,
                                      tick_interval=tick_interval,
                                      timerange=timerange)
