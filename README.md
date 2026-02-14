# data_preparation
Dataset Dimensions

Trades: 211,224 × 16

Sentiment: 2,644 × 4

Merged Dataset: 211,224 × 22

## Missing Values

1.No missing values detected in trading dataset

2.Duplicate Handling

3.Duplicate columns reviewed

4.Timestamp column normalized

5.Cleaned dataset prepared for analysis

## Data Cleaning & Alignment

1.Converted millisecond timestamps to datetime format

2.Extracted daily-level date (join_date)

3.Converted sentiment date column to datetime

4.Merged datasets using left join on daily date

5.Ensured daily alignment between trades and sentiment

## Key Metrics Created
## Daily PnL

Aggregated total Closed PnL per day

Identified daily profit and loss distribution

## Trader-Level Performance

1.Win Rate per account

2.Defined as % of trades with positive Closed PnL

3.Average Trade Size (USD)

4.Total Trades per account

5.Total PnL per account

## Leverage Distribution

1.Calculated leverage = Size USD / Start Position

2.Analyzed leverage spread across traders

3.Identified extreme leverage behavior

## Trading Activity

1.Number of trades per day

2.Daily trading volume trends

## Long / Short Ratio

Classified trades into:

1.Long

2.Short

3.Other
