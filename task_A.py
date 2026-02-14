import pandas as pd
import numpy as np

# STEP 1: LOAD THE DATASETS
# ==========================================
# Replace 'historical_data.csv' and 'fear_greed.csv' with your actual file names
df_trades = pd.read_csv('historical_data.csv')
df_sentiment = pd.read_csv('fear_greed.csv')

# ✅ SHOW ALL COLUMNS (keep row truncation as default)
pd.set_option('display.max_columns', None)

print("--- DATA LOADED ---")
print(f"Trades Shape: {df_trades.shape}")
print(f"Sentiment Shape: {df_sentiment.shape}")

# Documenting missing values (Task A.1)
print("\nMissing Values in Trades:\n", df_trades.isnull().sum())

# STEP 2: CLEAN & ALIGN DATES (Task A.2)
# ==========================================

# 1. Fix Trades Timestamp (The scientific notation column 'Timestamp')
# We use unit='ms' because your data is 1.73E+12 (milliseconds)
df_trades['datetime'] = pd.to_datetime(df_trades['Timestamp'], unit='ms')

# 2. Create a 'join_date' column (stripping the time) to match sentiment
df_trades['join_date'] = df_trades['datetime'].dt.normalize()

# 3. Fix Sentiment Date
# The image shows the column is named 'date' (e.g., 2/1/2018)
df_sentiment['join_date'] = pd.to_datetime(df_sentiment['date'])

# 4. Merge the datasets
# We assume we want to analyze the trades, so we 'left join' sentiment onto trades
df_merged = pd.merge(df_trades, df_sentiment, on='join_date', how='left')

print(f"\nMerged Data Shape: {df_merged.shape}")

# 
# STEP 3: CREATE KEY METRICS (Task A.3)
# 

# --- Metric 1: Leverage ---

#  We add a small number (1e-9) to avoid dividing by zero error if Start Position is 0.
df_merged['Calculated_Leverage'] = df_merged['Size USD'] / (df_merged['Start Position'] + 1e-9)
print("\n1. Leverage Distribution (Top 5000 rows):")
print(df_merged[['Account', 'Size USD', 'Start Position', 'Calculated_Leverage']].head(5000))

# --- Metric 2: Long/Short Ratio ---
# We use the unique values you showed in the image to classify direction.
def classify_side(direction):
    direction = str(direction).lower()
    if 'long' in direction or 'buy' in direction:
        return 'Long'
    elif 'short' in direction or 'sell' in direction:
        return 'Short'
    else:
        return 'Other'

df_merged['Trade_Side'] = df_merged['Direction'].apply(classify_side)

long_short_counts = df_merged['Trade_Side'].value_counts()
print("\n2. Long/Short Ratio:")
print(long_short_counts)

# --- Metric 3: Win Rate & Avg Trade Size ---
# We define a function to calculate stats for each trader
def calculate_trader_stats(x):
    total_trades = len(x)
    # A "Win" is when Closed PnL is positive
    wins = x[x['Closed PnL'] > 0]
    
    return pd.Series({
        'Win_Rate': len(wins) / total_trades if total_trades > 0 else 0,
        'Avg_Trade_Size_USD': x['Size USD'].mean(),
        'Total_Trades': total_trades,
        'Total_PnL': x['Closed PnL'].sum()
    })

# Group by Account to get stats per trader
trader_performance = df_merged.groupby('Account').apply(calculate_trader_stats)
print("\n3. Trader Performance (Sample):")
print(trader_performance.head(5000))

# --- Metric 4: Daily Activity ---
daily_stats = df_merged.groupby('join_date').agg({
    'Account': 'count',          # Number of trades per day
    'Closed PnL': 'sum'          # Daily PnL
}).rename(columns={'Account': 'Trade_Count', 'Closed PnL': 'Daily_PnL'})

print("\n4. Daily Activity (Sample):")
print(daily_stats.head())

# Final observation, duplicated column removing
print(df_merged.drop(columns=["timestamp"], inplace=True))

# Saving the file
# df_merged.to_csv("part__a__completed.csv", index=False)


#####   OUTPUT   #####
'''Trades Shape: (211224, 16)
Sentiment Shape: (2644, 4)

Missing Values in Trades: 
 Account             0    
Coin                0     
Execution Price     0     
Size Tokens         0     
Size USD            0     
Side                0     
Timestamp IST       0     
Start Position      0     
Direction           0     
Closed PnL          0     
Transaction Hash    0     
Order ID            0     
Crossed             0
Fee                 0
Trade ID            0
Timestamp           0
dtype: int64

Merged Data Shape: (211224, 22)

1. Leverage Distribution (Top 5000 rows):
                                         Account   Size USD  Start Position  \
0     0xae5eacaf9c6b9111fd53034a602c192a04e082ed    7872.16        0.000000
1     0xae5eacaf9c6b9111fd53034a602c192a04e082ed     127.68      986.524596
2     0xae5eacaf9c6b9111fd53034a602c192a04e082ed    1150.63     1002.518996
3     0xae5eacaf9c6b9111fd53034a602c192a04e082ed    1142.04     1146.558564
4     0xae5eacaf9c6b9111fd53034a602c192a04e082ed      69.75     1289.488521
...                                          ...        ...             ...
4995  0x513b8629fe877bb581bf244e326a047b249c4ff1    9999.59       23.667780
4996  0x513b8629fe877bb581bf244e326a047b249c4ff1   12432.15       23.788430
4997  0x513b8629fe877bb581bf244e326a047b249c4ff1    9999.59       23.938430
4998  0x513b8629fe877bb581bf244e326a047b249c4ff1   60028.22       24.059080
4999  0x513b8629fe877bb581bf244e326a047b249c4ff1  290942.15       24.783350

      Calculated_Leverage
0            7.872160e+12
1            1.294240e-01
2            1.147739e+00
3            9.960590e-01
4            5.409121e-02
...                   ...
4995         4.224980e+02
4996         5.226133e+02
4997         4.177212e+02
4998         2.495034e+03
4999         1.173942e+04

[5000 rows x 4 columns]

2. Long/Short Ratio:
Trade_Side
Long     115416
Short     95657
Other       151
Name: count, dtype: int64

3. Trader Performance (Sample):
                                            Win_Rate  Avg_Trade_Size_USD  \
Account
0x083384f897ee0f19899168e3b1bec365f52a9012  0.359612        16159.576734
0x23e7a7f8d14b550961925fbfdaa92f5d195ba5bd  0.442720         1653.226327
0x271b280974205ca63b716753467d5a371de622ab  0.301917         8893.000898
0x28736f43f1e871e6aa8b1148d38d4994275d72c4  0.438585          507.626933
0x2c229d22b100a7beb69122eed721cee9b24011dd  0.519914         3138.894782
0x3998f134d6aaa2b6a5f723806d00fd2bbbbce891  0.455215         1729.941104
0x39cef799f8b69da1995852eea189df24eb5cae3c  0.327668         4790.575486
0x3f9a0aadc7f04a7c9d75dc1b5a6ddd6e36486cf6  0.475904         3445.471265
0x420ab45e0bd8863569a5efbb9c05d91f40624641  0.234987         5189.367128
0x430f09841d65beb3f27765503d0f850b8bce7713  0.484236         2397.824753
0x47add9a56df66b524d5e2c1993a43cde53b6ed85  0.353445          517.528924
0x4acb90e786d897ecffb614dc822eb231b4ffb9f4  0.486226         9084.699093
0x4f93fead39b70a1824f981a54d4e55b278e9f760  0.360364        17098.171055
0x513b8629fe877bb581bf244e326a047b249c4ff1  0.401193        34396.580284
0x6d6a4b953f202f8df5bed40692e7fd865318264a  0.431795          746.725651
0x72743ae2822edd658c0c50608fd7c5c501b2afbd  0.345912         7216.667245
0x72c6a4624e1dffa724e6d00d64ceae698af892a0  0.306294         2133.667364
0x75f7eeb85dc639d5e99c78f95393aa9a5f1170d4  0.810876         2600.778049
0x7f4f299f74eec87806a830e3caa9afa5f2b9db8f  0.452213         3748.867511
0x8170715b3b381dffb7062c0298972d4727a0a63b  0.382743         2204.685531
0x8381e6d82f1affd39a336e143e081ef7620a3b7f  0.330194         6279.002287
0x8477e447846c758f5a675856001ea72298fd9cb5  0.261968          820.428513
0x92f17e8d81a944691c10e753af1b1baae1a2cd0d  0.285387         3601.689299
0xa0feb3725a9335f49874d7cd8eaad6be45b27416  0.345787         1273.195000
0xa520ded057a32086c40e7dd6ed4eb8efb82c00e0  0.573141         2066.689209
0xae5eacaf9c6b9111fd53034a602c192a04e082ed  0.408526         2979.441776
0xaf40fdc468c30116bd3307bcbf4a451a7ebf1deb  0.322097         8330.193371
0xb1231a4a2dd02f2276fa3c5e2a2f3436e6bfed23  0.337134         3837.885375
0xb899e522b5715391ae1d4f137653e7906c5e2115  0.438611        22504.555829
0xbaaaf6571ab7d571043ff1e313a9609a10637864  0.467582         3210.472831
0xbd5fead7180a9c139fa51a103cb6a2ce86ddb5c3  0.327527         7852.098338
0xbee1707d6b44d4d52bfe19e41f8a828645437aab  0.428230         1844.211886

                                            Total_Trades     Total_PnL
Account
0x083384f897ee0f19899168e3b1bec365f52a9012        3818.0  1.600230e+06
0x23e7a7f8d14b550961925fbfdaa92f5d195ba5bd        7280.0  4.788532e+04
0x271b280974205ca63b716753467d5a371de622ab        3809.0 -7.043619e+04
0x28736f43f1e871e6aa8b1148d38d4994275d72c4       13311.0  1.324648e+05
0x2c229d22b100a7beb69122eed721cee9b24011dd        3239.0  1.686580e+05
0x3998f134d6aaa2b6a5f723806d00fd2bbbbce891         815.0 -3.120360e+04
0x39cef799f8b69da1995852eea189df24eb5cae3c        3589.0  1.445692e+04
0x3f9a0aadc7f04a7c9d75dc1b5a6ddd6e36486cf6         332.0  5.349625e+04
0x420ab45e0bd8863569a5efbb9c05d91f40624641         383.0  1.995056e+05
0x430f09841d65beb3f27765503d0f850b8bce7713        1237.0  4.165419e+05
0x47add9a56df66b524d5e2c1993a43cde53b6ed85        8519.0  1.033437e+05
0x4acb90e786d897ecffb614dc822eb231b4ffb9f4        4356.0  6.777471e+05
0x4f93fead39b70a1824f981a54d4e55b278e9f760        7584.0  3.089759e+05
0x513b8629fe877bb581bf244e326a047b249c4ff1       12236.0  8.404226e+05
0x6d6a4b953f202f8df5bed40692e7fd865318264a         975.0  1.087312e+05
0x72743ae2822edd658c0c50608fd7c5c501b2afbd        1590.0  4.293556e+05
0x72c6a4624e1dffa724e6d00d64ceae698af892a0        1430.0  4.030115e+05
0x75f7eeb85dc639d5e99c78f95393aa9a5f1170d4        9893.0  3.790954e+05
0x7f4f299f74eec87806a830e3caa9afa5f2b9db8f        1559.0  1.490044e+04
0x8170715b3b381dffb7062c0298972d4727a0a63b        4601.0 -1.676211e+05
0x8381e6d82f1affd39a336e143e081ef7620a3b7f        1911.0  6.551366e+04
0x8477e447846c758f5a675856001ea72298fd9cb5       14998.0  4.391701e+04
0x92f17e8d81a944691c10e753af1b1baae1a2cd0d        3052.0  1.265789e+05
0xa0feb3725a9335f49874d7cd8eaad6be45b27416       15605.0  1.063029e+05
0xa520ded057a32086c40e7dd6ed4eb8efb82c00e0         417.0  7.284648e+04
0xae5eacaf9c6b9111fd53034a602c192a04e082ed         563.0  6.784562e+04
0xaf40fdc468c30116bd3307bcbf4a451a7ebf1deb         534.0  2.175883e+04
0xb1231a4a2dd02f2276fa3c5e2a2f3436e6bfed23       14733.0  2.143383e+06
0xb899e522b5715391ae1d4f137653e7906c5e2115        4838.0  2.248850e+04
0xbaaaf6571ab7d571043ff1e313a9609a10637864       21192.0  9.401638e+05
0xbd5fead7180a9c139fa51a103cb6a2ce86ddb5c3        2641.0  2.205191e+05
0xbee1707d6b44d4d52bfe19e41f8a828645437aab       40184.0  8.360806e+05

4. Daily Activity (Sample):
            Trade_Count     Daily_PnL
join_date
2023-03-28            3  0.000000e+00
2023-11-14         1045  1.555034e+02
2024-03-09         6962  1.769655e+05
2024-07-03         7141  1.587424e+05
2024-10-27        35241  3.189461e+06
'''