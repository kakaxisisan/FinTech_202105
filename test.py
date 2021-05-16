import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

df1 = pd.read_csv('data/train_v1.csv')
tmp1 = df1['amount'].groupby([df1['date'], df1['post_id'], df1['periods']]).sum()
df = pd.DataFrame(tmp1)
df.reset_index(inplace=True)
a_day = df[(df['date'] == '2019/8/3') & (df['post_id'] == 'B')]

# plt.plot(a_day['periods'], a_day['amount'])
# plt.show()

df_b = df1[df1['post_id']=='A'].drop(columns=['periods', 'biz_type']).drop_duplicates()
# print(df_b)
b1 = df_b.groupby(['date']).sum()
b2 = pd.DataFrame(b1)
b2.reset_index(inplace=True)
# amount_df = b2.amount
# amount_df.index=b2['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d'))
# amount_m = amount_df.resample('m').mean()
# amount_m.plot()
# plt.show()
# print(b2)
b2['year_month'] = b2['date'].apply(lambda x: '-'.join(x.split('/')[:2]+['1']))

tmp = pd.DataFrame(b2['amount'].groupby(b2['year_month']).mean())
tmp.reset_index(inplace=True)
tmp['time'] = pd.to_datetime(tmp['year_month'])
tmp = tmp.sort_values(by='time')
tmp = tmp.rename(columns={'amount': 'month_amount'})
# print(tmp)

tmp['month-1'] = tmp['month_amount'].shift(1)
tmp['month-2'] = tmp['month_amount'].shift(2)
tmp['month-3'] = tmp['month_amount'].shift(3)
tmp['month-4'] = tmp['month_amount'].shift(4)
tmp['month-5'] = tmp['month_amount'].shift(5)
tmp['month-6'] = tmp['month_amount'].shift(6)

b3 = b2.merge(tmp, on='year_month', how='left')
b3 = b3.drop(columns=['date', 'year_month', 'time'])

np.random.seed(0)
a = pd.DataFrame(np.random.randn(10, 3), index=np.arange(0, 10), columns=['q', 'a', 'b'])
plt.show()

