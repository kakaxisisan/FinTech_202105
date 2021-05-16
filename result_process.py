import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fun(x):
    return 0 if int(x) < 0 else int(x)

def get_periods_result():
    periods_df = pd.read_csv('result/result_test_v2_periods_trend_20210510_0.csv')
    day_df = pd.read_csv('data/test_v2_day.csv')
    periods_df['tmp'] = periods_df['amount'].apply(func=fun)
    df_final = periods_df.drop(columns='amount').rename(
            columns={'tmp': 'amount'})
    tmp1 = df_final['amount'].groupby([df_final['date'], df_final['post_id']]).sum()
    tmp2 = pd.DataFrame(tmp1)
    tmp2.reset_index(inplace=True)
    test_day = pd.merge(day_df, tmp2, on=['date', 'post_id'])
    test_day_ = test_day.drop(columns='amount_x').rename(
            columns={'amount_y': 'amount'})
    return df_final, test_day_

result_periods, result_day = get_periods_result()
print(result_periods)
print(result_day)
result_periods.to_csv('result/result_periods_v2_20210510.txt', sep=',', index=False)
result_day.to_csv('result/result_days_v2_20210510.txt', sep=',', index=False)


