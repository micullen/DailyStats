from typing import List, NamedTuple, Optional, Tuple, Any, Dict
import argparse
import re
import arrow

class TimeRange(NamedTuple):
    """
    NamedTuple Defining timerange inputs.
    [start/stop]type defines if [start/stop]ts shall be used.
    if *type is none, don't use corresponding startvalue.
    """
    starttype: Optional[str] = None
    stoptype: Optional[str] = None
    startts: int = 0
    stopts: int = 0


class Arguments(object):
    """
    Arguments Class. Manage the arguments received by the cli
    """

    def __init__(self, args: List[str], description: str) -> None:
        self.args = args
        self.parsed_arg: Optional[argparse.Namespace] = None
        self.parser = argparse.ArgumentParser(description=description)

    def _load_args(self) -> None:
        self.common_args_parser()
        self._build_subcommands()

    def get_parsed_arg(self) -> argparse.Namespace:
        """
        Return the list of arguments
        :return: List[str] List of arguments
        """
        if self.parsed_arg is None:
            self._load_args()
            self.parsed_arg = self.parse_args()

        return self.parsed_arg

    def parse_args(self) -> argparse.Namespace:
        """
        Parses given arguments and returns an argparse Namespace instance.
        """
        parsed_arg = self.parser.parse_args(self.args)

        return parsed_arg

    def common_args_parser(self) -> None:
        """
        Parses given common arguments and returns them as a parsed object.
        """
        self.parser.add_argument(
            '-v', '--verbose',
            help='verbose mode (-vv for more, -vvv to get all messages)',
            action='count',
            dest='loglevel',
            default=0,
        )
        self.parser.add_argument(
            '--version',
            action='version',
            version=f'%(prog)s {__version__}'
        )
        self.parser.add_argument(
            '-c', '--config',
            help='specify configuration file (default: %(default)s)',
            dest='config',
            default='config.json',
            type=str,
            metavar='PATH',
        )

    @staticmethod
    def parse_timerange(text: Optional[str]) -> TimeRange:
        """
        Parse the value of the argument --timerange to determine what is the range desired
        :param text: value from --timerange
        :return: Start and End range period
        """
        if text is None:
            return TimeRange(None, None, 0, 0)
        syntax = [(r'^-(\d{8})$', (None, 'date')),
                  (r'^(\d{8})-$', ('date', None)),
                  (r'^(\d{8})-(\d{8})$', ('date', 'date')),
                  (r'^-(\d{10})$', (None, 'date')),
                  (r'^(\d{10})-$', ('date', None)),
                  (r'^(\d{10})-(\d{10})$', ('date', 'date')),
                  (r'^(-\d+)$', (None, 'line')),
                  (r'^(\d+)-$', ('line', None)),
                  (r'^(\d+)-(\d+)$', ('index', 'index'))]
        for rex, stype in syntax:
            # Apply the regular expression to text
            match = re.match(rex, text)
            if match:  # Regex has matched
                rvals = match.groups()
                index = 0
                start: int = 0
                stop: int = 0
                if stype[0]:
                    starts = rvals[index]
                    if stype[0] == 'date' and len(starts) == 8:
                        start = arrow.get(starts, 'YYYYMMDD').timestamp
                    else:
                        start = int(starts)
                    index += 1
                if stype[1]:
                    stops = rvals[index]
                    if stype[1] == 'date' and len(stops) == 8:
                        stop = arrow.get(stops, 'YYYYMMDD').timestamp
                    else:
                        stop = int(stops)
                return TimeRange(stype[0], stype[1], start, stop)
        raise Exception('Incorrect syntax for timerange "%s"' % text)

    def testdata_dl_options(self) -> None:
        """
        Parses given arguments for testdata download
        """
        self.parser.add_argument(
            '--pairs-file',
            help='File containing a list of pairs to download',
            dest='pairs_file',
            default=None,
            metavar='PATH',
        )

        self.parser.add_argument(
            '--export',
            help='Export files to given dir',
            dest='export',
            default=None,
            metavar='PATH',
        )

        self.parser.add_argument(
            '--days',
            help='Download data for number of days',
            dest='days',
            type=int,
            metavar='INT',
            default=None
        )

        self.parser.add_argument(
            '--exchange',
            help='Exchange name (default: %(default)s)',
            dest='exchange',
            type=str,
            default='bittrex'
        )

        self.parser.add_argument(
            '-t', '--timeframes',
            help='Specify which tickers to download. Space separated list. \
                  Default: %(default)s',
            choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '3h', '4h',
                     '6h', '8h', '12h', '1d', '3d', '1w'],
            default=['1m', '5m'],
            nargs='+',
            dest='timeframes',
        )

        self.parser.add_argument(
            '--erase',
            help='Clean all existing data for the selected exchange/pairs/timeframes',
            dest='erase',
            action='store_true'
        )




