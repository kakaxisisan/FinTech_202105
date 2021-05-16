import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm

device = 'cpu'

def tmp():
    df1 = pd.read_csv('data/train_v1.csv', sep=',')
    df2 = pd.read_csv('data/wkd_v1.csv', sep=',').rename(columns={'ORIG_DT': 'date'})
    # print(df1)
    # df_total = pd.merge(df1, df2, on='date')
    # df_A = df_total[df_total['post_id'] == 'A'].drop(columns=['date', 'post_id'])
    wkd_dict = {'WN': 0, 'SN': 1, 'NH': 2, 'SS': 3, 'WS': 4}
    # df_A['tmp_1'] = df_A['WKD_TYP_CD'].map(wkd_dict)
    # df_A['tmp_2'] = df_A['biz_type'].str[1:].astype(int)
    # df_A_final = df_A.drop(columns=['WKD_TYP_CD', 'biz_type']).rename(columns={'tmp_1': 'wkd', 'tmp_2': 'type'})
    # tmp1 = df_A_final.group
    # print(df_A_final)
    # df_B = df_total[df_total['post_id'] == 'B'].drop(columns=['date', 'post_id', 'biz_type'])
    # df_B['tmp_1'] = df_B['WKD_TYP_CD'].map(wkd_dict)
    # df_B_final = df_B.drop(columns=['WKD_TYP_CD']).rename(columns={'tmp_1': 'wkd'})
    # print(df_B_final)
    # # print(len(df_A['amount'].unique()))

    tmp1 = df1['amount'].groupby([df1['date'], df1['post_id'], df1['periods']]).sum()
    # print(tmp1)
    tmp2 = pd.DataFrame(tmp1)
    # print(tmp2)
    tmp2.reset_index(inplace=True)
    # print(tmp2)
    count_a = (df1['post_id'] == 'A').astype(int).sum()
    count_b = (df1['post_id'] == 'B').astype(int).sum()
    ctb = (tmp2['post_id'] == 'B').astype(int).sum()
    cta = (tmp2['post_id'] == 'A').astype(int).sum()
    # print(df1['biz_type'].value_counts())
    # print(cta, ctb, count_b, count_a, count_b + count_a, count_a / 13 + count_b)
    # 合并A后一共99360条
    df_total = pd.merge(tmp2, df2, on='date')
    # print(df_total)
    post_dict = {'A': 0, 'B': 1}
    df_total['tmp_1'] = df_total['WKD_TYP_CD'].map(wkd_dict)
    df_total['tmp_2'] = df_total['post_id'].map(post_dict)
    df_final = df_total.drop(columns=['WKD_TYP_CD', 'post_id', 'date']).rename(
        columns={'tmp_1': 'wkd', 'tmp_2': 'post'})
    df_final = df_final[['post', 'periods', 'wkd', 'amount']]
    return df_final


def feature_process(all_features):
    # 特征处理
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features)
    return all_features


def get_train_set_with_trend():
    # 得到训练集和测试集
    df1 = pd.read_csv('data/train_v2.csv', sep=',')
    df3 = pd.read_csv('data/test_v2_periods.csv', sep=',')
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

    df_final['year_month'] = df_final['date'].apply(lambda x: '-'.join(x.split('/')[:2] + ['1']))
    test_final = df_final[train_num:]
    df_final = df_final[: train_num]

    tmp = pd.DataFrame(df_final['amount'].groupby(df_final['year_month']).mean())
    tmp.reset_index(inplace=True)
    tmp['time'] = pd.to_datetime(tmp['year_month'])
    tmp = tmp.sort_values(by='time')
    tmp = tmp.rename(columns={'amount': 'month_amount'})
    tmp['month_amount'] = tmp['month_amount'].astype(int)
    tmp = tmp.append({'year_month': '2020-12-1', 'month_amount': 0.0, 'time': '2020-12-01'}, ignore_index=True)

    tmp['month-1'] = tmp['month_amount'].shift(1)
    tmp['month-2'] = tmp['month_amount'].shift(2)
    tmp['month-3'] = tmp['month_amount'].shift(3)
    tmp['month-4'] = tmp['month_amount'].shift(4)
    tmp['month-5'] = tmp['month_amount'].shift(5)
    tmp['month-6'] = tmp['month_amount'].shift(6)

    b3 = df_final.merge(tmp, on='year_month', how='left')
    b3 = b3.drop(columns=['date', 'year_month', 'time']).rename(columns={'month_amount': 'month_0'}).dropna()
    df_final = b3[['post_id', 'periods', 'wkd', 'month_0', 'month-1',
                   'month-2', 'month-3', 'month-4', 'month-5', 'month-6', 'amount']]

    b4 = test_final.merge(tmp, on='year_month', how='left')
    b4 = b4.drop(columns=['date', 'year_month', 'time']).rename(columns={'month_amount': 'month_0'})
    b4['amount'] = b4['amount'].fillna(0)
    test_final = b4[['post_id', 'periods', 'wkd', 'month_0', 'month-1',
                     'month-2', 'month-3', 'month-4', 'month-5', 'month-6', 'amount']]

    all_train_features = df_final.iloc[:, :-1]
    all_test_features = test_final.iloc[:, :-1]
    train_data = feature_process(all_train_features)
    test_data = feature_process(all_test_features)
    # print(all_features)
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_data, df_final.iloc[:, -1], test_size=0.2, random_state=20210504, stratify=df_final['post_id'])
    return train_features, val_features, train_labels, val_labels, test_data


def get_train_set():
    df1 = pd.read_csv('data/train_v1.csv', sep=',')
    df2 = pd.read_csv('data/wkd_v1.csv', sep=',').rename(columns={'ORIG_DT': 'date'})
    tmp1 = df1['amount'].groupby([df1['date'], df1['post_id'], df1['periods']]).sum()
    tmp2 = pd.DataFrame(tmp1)
    tmp2.reset_index(inplace=True)
    df_total = pd.merge(tmp2, df2, on='date')
    wkd_dict = {'WN': 0, 'SN': 1, 'NH': 2, 'SS': 3, 'WS': 4}
    df_total['tmp_1'] = df_total['WKD_TYP_CD'].map(wkd_dict)
    df_final = df_total.drop(columns=['WKD_TYP_CD', 'date']).rename(
        columns={'tmp_1': 'wkd'})
    df_final = df_final[['post_id', 'periods', 'wkd', 'amount']]
    all_features = df_final.iloc[:, :-1]
    train_data = feature_process(all_features)
    # print(all_features)
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_data, df_final.iloc[:, -1], test_size=0.2, random_state=20210504, stratify=df_final['post_id'])
    return train_features, val_features, train_labels, val_labels


def get_test_set():
    df1 = pd.read_csv('data/test_v1_periods.csv', sep=',')
    df2 = pd.read_csv('data/wkd_v1.csv', sep=',').rename(columns={'ORIG_DT': 'date'})
    df_total = pd.merge(df1, df2, on='date')
    wkd_dict = {'WN': 0, 'SN': 1, 'NH': 2, 'SS': 3, 'WS': 4}
    df_total['tmp_1'] = df_total['WKD_TYP_CD'].map(wkd_dict)
    df_final = df_total.drop(columns=['WKD_TYP_CD', 'date']).rename(
        columns={'tmp_1': 'wkd'})
    df_final = df_final[['post_id', 'periods', 'wkd', 'amount']]
    all_features = df_final.iloc[:, :-1]
    all_features = feature_process(all_features)
    return all_features


class myDataset:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx, :], self.label[idx]


# 转化为tensor
# train_features, val_features, train_labels, val_labels = get_train_set()
# test_data = get_test_set()
train_features, val_features, train_labels, val_labels, test_data = get_train_set_with_trend()
# 训练集
# print(train_features.values)
train_features = torch.tensor(train_features.values, dtype=torch.float)
# 验证集
val_features = torch.tensor(val_features.values, dtype=torch.float)
#  测试集
test_features = torch.tensor(test_data.values, dtype=torch.float)

train_labels = torch.tensor(train_labels.values, dtype=torch.float)
train_labels = train_labels.unsqueeze(1)

val_labels = torch.tensor(val_labels.values, dtype=torch.float)
val_labels = val_labels.unsqueeze(1)

print(f'训练集数据: {train_features.shape}')
print(f'验证集数据: {val_features.shape}')
print(f'测试集数据: {test_features.shape}')

train_dataset = myDataset(train_features, train_labels)
val_dataset = myDataset(val_features, val_labels)

# 变为迭代器
train_iter = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_iter = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=4)

def _weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# 网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.apply(_weight_init)  # 初始化参数

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 使用mape作为自定义得分函数，这也是比赛的判定标准
def custom_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    return mape


def train(net, data_iter, phase, criterion, optimizer=None):
    y_true = []
    y_pred = []
    mean_loss = []
    is_grad = True if phase == 'train' else False
    with torch.set_grad_enabled(is_grad):
        net.train()
        for step, (X, y) in enumerate(data_iter):
            X = X.to(device)
            y = y.to(device)
            out = net(X)
            loss = criterion(out, y)  # 计算损失
            mean_loss.append(loss.item())

            if phase == 'train':
                optimizer.zero_grad()  # optimizer 0
                loss.backward()  # back propragation
                optimizer.step()  # update the paramters

            # 将每一个step的结果加入列表，最后统一生产这一个epoch的指标
            # 添加预测值和真实类标签
            y_pred.extend(out.detach().cpu().squeeze().numpy().tolist())
            y_true.extend(y.detach().cpu().squeeze().numpy().tolist())

    # 全量样本的rmse和平均loss
    rmse = custom_score(y_true, y_pred)
    mean_loss = np.mean(mean_loss)
    # 保留4位小数
    rmse = np.round(rmse, 4)
    mean_loss = np.round(mean_loss, 4)
    return mean_loss, rmse



net = Net()
criterion = torch.nn.MSELoss()  # 损失函数为MSE
net = net.to(device)  # 将网络和损失函数转化为GPU或CPU
criterion = criterion.to(device)
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.005, weight_decay=0)

epochs = 500
print(f'{datetime.now()} 开始训练...')
for epoch in tqdm(range(epochs)):
    train_mean_loss, train_score = train(net=net,
                                         data_iter=train_iter,
                                         phase='train',
                                         criterion=criterion,
                                         optimizer=optimizer)

    val_mean_loss, val_score = train(net=net,
                                     data_iter=train_iter,
                                     phase='val',
                                     criterion=criterion,
                                     optimizer=None)
    if epoch % 10 == 0:
        tqdm.write(f'Epoch: {epoch} Train loss: {train_mean_loss} Val loss: {val_mean_loss}', end=' ')
        tqdm.write(f'Train score: {train_score} Val score: {val_score}')


print(f'{datetime.now()} 训练结束...')

submission = pd.read_csv('data/test_v2_periods.csv')
predict = net(test_features)
predict = predict.detach().squeeze().numpy()
submission['amount'] = predict
submission.to_csv('result/result_test_v2_periods_trend_20210510_0.csv', index=False)

