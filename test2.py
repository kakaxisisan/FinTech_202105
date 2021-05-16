import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


df1 = pd.read_csv('data/train_v1.csv', sep=',')
df3 = pd.read_csv('data/test_v1_periods.csv', sep=',')
df2 = pd.read_csv('data/wkd_v1.csv', sep=',').rename(columns={'ORIG_DT': 'date'})
tmp1 = df1['amount'].groupby([df1['date'], df1['post_id'], df1['periods']]).sum()
tmp2 = pd.DataFrame(tmp1)
tmp2.reset_index(inplace=True)
train_num = tmp2.shape[0]
df4 = pd.concat([tmp2, df3])

df_total = pd.merge(df4, df2, on='date')
wkd_dict = {'WN': 0, 'SN': 1, 'NH': 2, 'SS': 3, 'WS': 4}
df_total['tmp_1'] = df_total['WKD_TYP_CD'].map(wkd_dict)
df_final = df_total.drop(columns=['WKD_TYP_CD']).rename(
    columns={'tmp_1': 'wkd'})

df_final['year_month'] = df_final['date'].apply(lambda x: '-'.join(x.split('/')[:2]+['1']))
test_final = df_final[train_num:]
df_final = df_final[: train_num]

tmp = pd.DataFrame(df_final['amount'].groupby(df_final['year_month']).mean())
tmp.reset_index(inplace=True)
tmp['time'] = pd.to_datetime(tmp['year_month'])
tmp = tmp.sort_values(by='time')
tmp = tmp.rename(columns={'amount': 'month_amount'})
tmp['month_amount'] = tmp['month_amount'].astype(int)
tmp = tmp.append({'year_month': '2020-11-1', 'month_amount': float(0), 'time': '2020-11-01'}, ignore_index=True)

tmp['month-1'] = tmp['month_amount'].shift(1)
tmp['month-2'] = tmp['month_amount'].shift(2)
tmp['month-3'] = tmp['month_amount'].shift(3)
tmp['month-4'] = tmp['month_amount'].shift(4)
tmp['month-5'] = tmp['month_amount'].shift(5)
tmp['month-6'] = tmp['month_amount'].shift(6)

b3 = df_final.merge(tmp, on='year_month', how='left')
date_df = b3['month_amount'].rolling(window=200, min_periods=1).mean()
print(date_df)
date_df.plot()
plt.show()

b3 = b3.drop(columns=['date', 'year_month', 'time']).rename(columns={'month_amount': 'month_0'}).dropna()
df_final = b3[['post_id', 'periods', 'wkd', 'month_0', 'month-1',
       'month-2', 'month-3', 'month-4', 'month-5', 'month-6', 'amount']]


b4 = test_final.merge(tmp, on='year_month', how='left')
b4 = b4.drop(columns=['date', 'year_month', 'time']).rename(columns={'month_amount': 'month_0'})
b4['amount'] = b4['amount'].fillna(0)
test_final = b4[['post_id', 'periods', 'wkd', 'month_0', 'month-1',
       'month-2', 'month-3', 'month-4', 'month-5', 'month-6', 'amount']]

