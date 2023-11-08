

import numpy as np
import pandas as pd
import os, sys, re, ast, csv, math, gc, random, enum, argparse, json, requests, time  
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None) # to ensure console display all columns
pd.set_option('display.float_format', '{:0.3f}'.format)
pd.set_option('display.max_row', 50)
plt.style.use('ggplot')
from pathlib import Path
import joblib
from copy import deepcopy

root_folder = "."
projectPath = Path(rf'{root_folder}')

dataPath = projectPath / 'data'
pickleDataPath = dataPath / 'pickle'
htmlDataPath = dataPath / 'html'
imageDataPath = dataPath / 'image'
dataInputPath = dataPath / 'input'
dataWorkingPath = dataPath / 'working'
dataOutputPath = dataPath / 'output'
modelPath = projectPath / 'models'
configPath = projectPath / 'config'

dataInputPath.mkdir(parents=True, exist_ok=True)

import pickle
def save_obj(obj, name):
    with open(pickleDataPath / f'{name}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(pickleDataPath / f'{name}.pkl', 'rb') as f:
        return pickle.load(f)






##############################################################################
## Imports


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from src.utils import (
    load_config,
)

from src.project_functions import (
    get_profit_accuracy,
    get_threshold,
)


def set_seed(seed: int = 100) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)



def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc




def get_rnn_dataloader_from_array(X_data, y_data, window_size, batch_size, is_test_loader=False):
    X, y = [], []
    for i in range(window_size, len(X_data)+1):
        feature = X_data[(i-window_size):i,:]
        target = y_data[i-1]
        X.append(feature)
        y.append(target)
    X = torch.tensor(X).float()
    y = torch.tensor(y).long()
    if is_test_loader:
        data_loader = DataLoader(TensorDataset(X, y), batch_size=1)
    else:
        data_loader = DataLoader(TensorDataset(X, y), shuffle=True, batch_size=batch_size)
    return data_loader


def get_torch_rnn_dataloaders(col_feature, col_target, train_df, valid_df, test_df, window_size, batch_size):

    X_train = train_df[col_feature].values
    y_train = train_df[col_target].values

    X_val = valid_df[col_feature].values
    y_val = valid_df[col_target].values

    X_test = test_df[col_feature].values
    y_test = test_df[col_target].values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    train_loader = get_rnn_dataloader_from_array(X_train, y_train, window_size, batch_size, is_test_loader=False)
    val_loader = get_rnn_dataloader_from_array(X_val, y_val, window_size, batch_size, is_test_loader=False)
    test_loader = get_rnn_dataloader_from_array(X_test, y_test, window_size, batch_size, is_test_loader=True)

    return (train_loader, val_loader, test_loader)





def t2v(tau, f, out_features, w, b, w0, b0):
    v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)



class GRUmodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_t2v=True):
        super(GRUmodel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_t2v = use_t2v

        if self.use_t2v:
            self.t2v_layer = SineActivation(in_features=input_size, out_features=16)
            self.layer0 = nn.Linear(16, input_size)

        self.recurrent_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.layer1 = nn.Linear(hidden_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, output_size)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        if self.use_t2v:
            x = self.t2v_layer(x)
            x = self.layer0(x)

        o, h = self.recurrent_layer(x)
        h = h.squeeze().unsqueeze(0) if len(h.squeeze().shape) < 2 else h.squeeze()
        x = self.layer1(h)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        output = self.layer3(x)
        output if len(output.shape) > 1 else output.unsqueeze(0)
        return output





def get_rnn_model(
    col_feature, train_loader, val_loader,
    epochs, batch_size, learning_rate, window_size, hidden_size,
):

    input_size = len(col_feature)
    output_size = 3
    criterion = nn.CrossEntropyLoss()
    model = GRUmodel(input_size, hidden_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    for e in range(1, epochs+1):
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            if X_train_batch.shape[0] == 1:
                continue

            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        model.eval()
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))

    return model



def get_predictions(test_loader, model):
    y_pred_list = []
    y_score_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_score, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_score_list.append(y_pred_score.cpu().numpy())
            y_pred_list.append(y_pred_tags.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_score_list = [a.squeeze().tolist() for a in y_score_list]
    return (y_pred_list, y_score_list)




def run_mlp_pipeline(
    col_feature, col_target, train_df, valid_df, test_df, 
    epochs, batch_size, learning_rate, window_size, hidden_size,
):
    set_seed(100)
    (train_loader, val_loader, test_loader) = get_torch_rnn_dataloaders(
        col_feature, col_target, train_df, valid_df, test_df, window_size, batch_size,
    )
    model = get_rnn_model(
        col_feature, train_loader, val_loader,epochs, batch_size, learning_rate, window_size, hidden_size,
    )
    (y_pred_list, y_score_list) = get_predictions(test_loader, model)

    return (y_pred_list, y_score_list)




##############################################################################
## Settings

config = load_config(path=configPath / "settings.yml")

col_date = ['datetime']
col_price = 'close_D1'
col_target = 'target'
start_year = 2013
prediction_lookforward_days = 1
col_target_gains = f'gains_N{prediction_lookforward_days}D'




##############################################################################
## Main


ref_ticker = 'CL'
text_data = load_config(path=configPath / f"text_data_{ref_ticker}.yml")

cols = [
    'datetime', 'close_D1', 'price', 
    'D1-SMA10-val', 'D1-MACD-macd', 'D1-MACD-macdsignal', 'D1-MACD-macdhist', 'D1-ROC2-val',
    'D1-MOM4-val', 'D1-RSI10-val', 'D1-BB20-upper', 'D1-BB20-mid', 'D1-BB20-lower', 'D1-CCI20-val',
    'D1-PSAR-val',
    'D1-VMD-c0', 'D1-VMD-c1', 'D1-VMD-c2',
    'D1-VMD-c3', 'D1-VGD60-c0', 'D1-VGD60-c1', 'D1-VGD60-c2', 'D1-VGD60-c3',
    'D1-VGD120-c0', 'D1-VGD120-c1', 'D1-VGD120-c2', 'D1-VGD120-c3',
    'D1-VGC60-c0', 'D1-VGC60-c1', 'D1-VGC60-c2', 'D1-VGC60-c3',
    'D1-VGC120-c0', 'D1-VGC120-c1', 'D1-VGC120-c2', 'D1-VGC120-c3',
    'D1-VGH60-c0', 'D1-VGH60-c1', 'D1-VGH60-c2', 'D1-VGH60-c3',
    'D1-VGH120-c0', 'D1-VGH120-c1', 'D1-VGH120-c2', 'D1-VGH120-c3', 
    'VMD_0', 'VMD_1', 'VMD_2', 'VMD_3',
    'VG_degree_60_0', 'VG_closeness_60_0', 'VG_harmonic_60_0', 
    'VG_degree_60_1', 'VG_closeness_60_1', 'VG_harmonic_60_1', 
    'VG_degree_60_2', 'VG_closeness_60_2', 'VG_harmonic_60_2', 
    'VG_degree_120_0', 'VG_closeness_120_0', 'VG_harmonic_120_0', 
    'VG_degree_120_1', 'VG_closeness_120_1', 'VG_harmonic_120_1', 
    'VG_degree_120_2', 'VG_closeness_120_2', 'VG_harmonic_120_2', 
    'us_oil_supply', 'oil_volatility', 'oil_global', 'oil_gas_ep', 'dff', 'cpi_usa',
    'gains_N1D',
]


file = open(dataInputPath / f"data.csv","w")
file.write(",".join(cols)+"\n")
wrt = []
for _year in [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
    wrt += text_data[_year]

file.write("\n".join(wrt))
file.close()


data_df = pd.read_csv(dataInputPath / f"data.csv")
data_df['datetime'] = data_df['datetime'].apply(lambda t: datetime.utcfromtimestamp(int(t)/1000))
data_df['year'] = data_df['datetime'].dt.year
data_df['month'] = data_df['datetime'].dt.month
data_df['year_month'] = data_df['year'].astype(str) + "-" + data_df['month'].astype(str).apply(lambda s: s.zfill(2))
data_df.sort_values('datetime', ascending=True, inplace=True)
data_df.reset_index(drop=True, inplace=True)



## Feature Settings
feature_map_dict = {
    "VMD": ['close_D1', 'VMD_0','VMD_1','VMD_2','VMD_3'],
    # combi: 0.558, 96 / 1,482

    "VGH60": ['close_D1', 'VG_harmonic_60_0', 'VG_harmonic_60_1', 'VG_harmonic_60_2'],
    # combi: 0.594, 41 / 1,482

    "VGH120": ['close_D1', 'VG_harmonic_120_0', 'VG_harmonic_120_1', 'VG_harmonic_120_2'],
    # combi: 0.615, 32 / 1,482

    "VGC60": ['close_D1', 'VG_closeness_60_0', 'VG_closeness_60_1', 'VG_closeness_60_2'],
    # combi: 0.538, 35 / 1,482

    "VGD60": ['close_D1', 'VG_degree_60_0', 'VG_degree_60_1', 'VG_degree_60_2'],
    # combi: 0.562, 59 / 1,482

    # "BOTH": [
    #     'close_D1',
    #     'D1-SMA10-val',
    #     'D1-BB20-upper',
    #     'D1-BB20-mid',
    #     'D1-BB20-lower',
    #     'D1-ROC2-val',
    #     'D1-MOM4-val',
    #     'D1-RSI10-val',
    #     'D1-CCI20-val',
    #     'D1-PSAR-val',
    # 
    #     'us_oil_supply',
    # ],

    # "TECH": [
    #     'close_D1',
    #     'D1-SMA10-val',
    #     'D1-BB20-upper',
    #     'D1-BB20-mid',
    #     'D1-BB20-lower',
    #     'D1-ROC2-val',
    #     'D1-MOM4-val',
    #     'D1-RSI10-val',
    #     'D1-CCI20-val',
    #     'D1-PSAR-val',
    # ],
    # combi: 0.533, 98 / 1,482

    "FUND": [
        'close_D1',
        'us_oil_supply',
        'oil_volatility',
        'oil_gas_ep',

        'oil_global',

        'dff',
        'cpi_usa',
    ],

}
# combi: 0.583, 211 / 1,482



## General GRU Settings (With T2V)
epochs = 1
hidden_size = 50
window_size = 5
batch_size = 8
thres_multiplier = 3
learning_rate = 0.0005
threshold_stepsize = 0.01
valid_lookback_months = 12
train_lookback_months = 48

result_dict = {}
test_df_full = pd.DataFrame()


idx_test = data_df['year'].isin([2018, 2019, 2020, 2021, 2022, 2023])
year_month_list = sorted(list(set(data_df['year_month'])))
test_year_month_list = sorted(list(set(data_df.loc[idx_test,'year_month'])))
year_month_vec = np.array(year_month_list)

for test_year_month in test_year_month_list:
    print(f"##  test_year_month: {test_year_month}")

    test_year = int(test_year_month.split('-')[0])
    test_month = int(test_year_month.split('-')[1])

    valid_year_month_list = list(year_month_vec[year_month_vec < test_year_month][-valid_lookback_months:])
    train_year_month_list = list(year_month_vec[year_month_vec < min(valid_year_month_list)][-train_lookback_months:])
    test_year_month_list = [test_year_month]

    result_dict[test_year_month] = {}
    data_df_temp = data_df.copy()

    thres_df = data_df_temp[data_df_temp['year_month'].isin(train_year_month_list)].copy()
    thres_df.reset_index(drop=True, inplace=True)
    col_target_gains_thres = get_threshold(thres_df[col_price], stepsize=threshold_stepsize) * thres_multiplier
    data_df_temp[col_target] = 1
    data_df_temp.loc[data_df_temp[col_target_gains] < -col_target_gains_thres, col_target] = 0
    data_df_temp.loc[data_df_temp[col_target_gains] > col_target_gains_thres, col_target] = 2

    pred_df = data_df_temp[data_df_temp['year_month'].isin(test_year_month_list)].copy()
    pred_df.reset_index(drop=True, inplace=True)

    for feature_type in feature_map_dict:
        col_feature = feature_map_dict[feature_type].copy()
        model_df = data_df_temp[['datetime','year_month',col_target]+col_feature].copy()
        model_df.reset_index(drop=True, inplace=True)

        model_df = model_df.dropna()
        model_df.reset_index(drop=True, inplace=True)

        train_df = model_df[model_df['year_month'].isin(train_year_month_list)].copy()
        valid_df = model_df[model_df['year_month'].isin(valid_year_month_list)].copy()
        test_df = model_df[model_df['year_month'].isin(test_year_month_list)].copy()

        valid_df_windowed = pd.concat([train_df,valid_df]).copy()
        valid_df_windowed = valid_df_windowed.tail(len(valid_df) + window_size-1)

        test_df_windowed = pd.concat([train_df, valid_df, test_df]).copy()
        test_df_windowed = test_df_windowed.tail(len(test_df) + window_size-1)

        (y_pred_list, y_score_list) = run_mlp_pipeline(
            col_feature, col_target, train_df, valid_df_windowed, test_df_windowed,
            epochs, batch_size, learning_rate, window_size, hidden_size,
            )
        test_df[f'pred_{feature_type}'] = y_pred_list
        test_df[f'score_{feature_type}'] = y_score_list

        test_df.reset_index(drop=True, inplace=True)
        merge_df = pd.merge(pred_df['datetime'], test_df, how='left', on='datetime')
        pred_df[f'pred_{feature_type}'] = merge_df[f'pred_{feature_type}'].values
        pred_df[f'pred_{feature_type}'] = pred_df[f'pred_{feature_type}'].fillna(1)


    def get_pred_combi(row, feature_map_dict):
        pred_list = [(row[f'pred_{feature_type}']-1) for feature_type in feature_map_dict]
        pred_list = [x for x in pred_list if x != 0]
        if len(pred_list) == 0:
            out = 1
        else:
            out = int(np.sign(np.mean(pred_list))) + 1
        return out

    pred_df['pred_combi'] = pred_df.apply(get_pred_combi, axis=1, feature_map_dict=feature_map_dict)

    result_dict[test_year_month]['prediction_data'] = pred_df.copy()
    test_df_full = pd.concat([test_df_full, pred_df])

    for pred_type in ['combi']:
        col_pred = f'pred_{pred_type}'
        (profit_accuracy, pred_count, total_count, short_accuracy, long_accuracy) = get_profit_accuracy(pred_df, col_pred, col_target_gains)
        print(f"# {pred_type}: {profit_accuracy:,.3f}, {pred_count:,.0f} out of {total_count:,.0f}")
        result_dict[test_year_month][pred_type] = {}
        result_dict[test_year_month][pred_type]['profit_accuracy'] = profit_accuracy
        result_dict[test_year_month][pred_type]['pred_count'] = pred_count
        result_dict[test_year_month][pred_type]['total_count'] = total_count

    print()


pred_type = 'combi'
col_pred = f'pred_{pred_type}'

plot_df = test_df_full.copy()
# plot_df = plot_df.tail(2000).head(500)
plot_df.reset_index(drop=True, inplace=True)
# plot_df['plot_time'] = plot_df['datetime'].apply(lambda d: d.strftime('%Y-%m-%d %H:%M:%S'))
plot_df['SMA'] = plot_df[col_price].rolling(100).mean()

idx = (plot_df['price'] > plot_df['SMA'])
idx = idx & (plot_df[col_pred] == 0)
plot_df.loc[idx, col_pred] = 1

idx = (plot_df['price'] < plot_df['SMA'])
idx = idx & (plot_df[col_pred] == 2)
plot_df.loc[idx, col_pred] = 1

(profit_accuracy, pred_count,
 total_count,
 short_accuracy, long_accuracy) = get_profit_accuracy(plot_df, col_pred, col_target_gains)

plt.plot(plot_df['datetime'], plot_df[col_price], color='gray')
plt.plot(plot_df['datetime'], plot_df['SMA'], color='purple')

idx = plot_df[col_pred] == 2
plt.scatter(plot_df.loc[idx, 'datetime'], plot_df.loc[idx, col_price], color='blue', alpha=0.5)
idx = plot_df[col_pred] == 0
plt.scatter(plot_df.loc[idx, 'datetime'], plot_df.loc[idx, col_price], color='red', alpha=0.5)
# plt.xticks(rotation=45)
# plt.xticks(visible=False)
# plt_title = f"{pred_type}: {profit_accuracy:,.3f}, {long_accuracy:,.3f}, {short_accuracy:,.3f}, {pred_count:,.0f} / {total_count:,.0f}"


plt_title = f"{pred_type}: {profit_accuracy:,.3f}, {pred_count:,.0f} / {total_count:,.0f}"
plt.title(plt_title)

for test_year in [2018, 2019, 2020, 2021, 2022, 2023]:
    col_pred = f'pred_{pred_type}'
    (profit_accuracy, pred_count,
     total_count,
     short_accuracy, long_accuracy) = get_profit_accuracy(plot_df[plot_df['year'] == test_year], col_pred, col_target_gains)
    plt.text(datetime(test_year,1,1), 10, f"{profit_accuracy:,.3f}")
plt.show()
print(f"# {plt_title}")









