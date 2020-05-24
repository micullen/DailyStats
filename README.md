# DailyStats

This repository retrieves market data for XBT and produces results which can lend a hand for a daily bias for trading. Relative Volume/Range and Sharpe ratio of statistics and signals.

<ins>Relative Volume/Range</ins>
Due to price typically following Range -> Expansion -> Range etc, low volume and range percentiles (<20%) can indicate that expansion is likely to occur and a trending strategy should be taken. High volume and range percentiles (>80%) can indicate that price is likely to range following the current trending environment, so mean reversion strategies should begin to be replaced by trending strategies.

<ins>Sharpe Ratio of Signals and Stats</ins>
The more signals that are active and display in the table (with high sharpe ratio), the more bullish price is. i.e. price > weekly vwap, daily returns are more often positive than not, so they can lend a helpful bias for intraday trades.

## Usage

```python
python DailyStats/createdailyreport.py

```

## Requirements
pandas

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
