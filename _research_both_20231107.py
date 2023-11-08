

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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import networkx as nx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from src.utils import (
    load_config,
)

from src.project_functions import (
    get_profit_accuracy,
    get_threshold,
    VMD,
    visibility_graph,
)


def set_seed(seed: int = 100) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")

def get_class_distribution(obj):
    count_dict = {
        0: 0,
        1: 0,
        2: 0,
    }
    for i in obj:
        count_dict[i] += 1
    return count_dict


def get_weighted_sampler(y):
    target_list = []
    for t in y:
        target_list.append(t)
    target_list = torch.tensor(target_list)

    class_count = [i for i in get_class_distribution(target_list.cpu().numpy()).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    return (weighted_sampler, class_weights)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True




def get_rnn_dataloader_from_array(X_data, y_data, window_size, batch_size, is_test_loader=False, use_weighted_sampler=False):

    weighted_sampler = None
    class_weights = None
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
        if use_weighted_sampler:
            # (weighted_sampler, class_weights) = get_weighted_sampler(list(y_data), len(y))
            (weighted_sampler, class_weights) = get_weighted_sampler(y)
            data_loader = DataLoader(TensorDataset(X, y), sampler=weighted_sampler, batch_size=batch_size)
        else:
            data_loader = DataLoader(TensorDataset(X, y), shuffle=True, batch_size=batch_size)

    return (data_loader, weighted_sampler, class_weights)




def get_torch_rnn_dataloaders(col_feature, col_target, train_df, valid_df, test_df, window_size, batch_size, use_weighted_sampler=False):

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

    (train_loader, weighted_sampler, class_weights) = get_rnn_dataloader_from_array(X_train, y_train, window_size, batch_size, is_test_loader=False, use_weighted_sampler=use_weighted_sampler)
    (val_loader, _, _) = get_rnn_dataloader_from_array(X_val, y_val, window_size, batch_size, is_test_loader=False, use_weighted_sampler=False)
    (test_loader, _, _) = get_rnn_dataloader_from_array(X_test, y_test, window_size, batch_size, is_test_loader=True)

    return (train_loader, val_loader, test_loader, weighted_sampler, class_weights)









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

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)



class GRUmodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, arc_num=1, use_t2v=True):
        super(GRUmodel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.arc_num = arc_num
        self.use_t2v = use_t2v

        if self.use_t2v:
            self.t2v_layer = SineActivation(in_features=input_size, out_features=16)
            self.layer0 = nn.Linear(16, input_size)

        self.recurrent_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        if self.arc_num == 1:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.layer3 = nn.Linear(256, output_size)

        if self.arc_num == 2:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.layer3 = nn.Linear(256, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.layer4 = nn.Linear(32, output_size)

        if self.arc_num == 3:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.layer3 = nn.Linear(64, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.layer4 = nn.Linear(32, output_size)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        if self.use_t2v:
            x = self.t2v_layer(x)
            x = self.layer0(x)

        o, h = self.recurrent_layer(x)
        h = h.squeeze().unsqueeze(0) if len(h.squeeze().shape) < 2 else h.squeeze()

        if self.arc_num == 1:
            x = self.layer1(h)
            x = self.bn1(x)
            x = self.layer2(x)
            x = self.bn2(x)
            output = self.layer3(x)


        if self.arc_num in [2,3]:
            x = self.layer1(h)
            x = self.bn1(x)
            x = self.layer2(x)
            x = self.bn2(x)
            x = self.layer3(x)
            x = self.bn3(x)
            output = self.layer4(x)

        output if len(output.shape) > 1 else output.unsqueeze(0)

        return output




class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 use_dual_lstm=False, arc_num=0, use_t2v=True):
        super(LSTMmodel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_dual_lstm = use_dual_lstm
        self.arc_num = arc_num
        self.use_t2v = use_t2v

        if self.use_t2v:
            self.t2v_layer = SineActivation(in_features=input_size, out_features=16)
            self.layer0 = nn.Linear(16, input_size)

        self.recurrent_layer1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        if self.use_dual_lstm:
            self.recurrent_layer2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        if self.arc_num == 0:
            self.layer1 = nn.Linear(hidden_size, output_size)

        if self.arc_num == 1:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.layer3 = nn.Linear(256, output_size)

        if self.arc_num == 2:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.layer3 = nn.Linear(256, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.layer4 = nn.Linear(32, output_size)

        if self.arc_num == 3:
            self.layer1 = nn.Linear(hidden_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.layer2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.layer3 = nn.Linear(64, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.layer4 = nn.Linear(32, output_size)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        if self.use_t2v:
            x = self.t2v_layer(x)
            x = self.layer0(x)    

        rx, (hn, cn) = self.recurrent_layer1(x)
        if self.use_dual_lstm:
            rx, (hn, cn) = self.recurrent_layer2(rx)

        if self.arc_num == 0:
            output = self.layer1(rx[:,-1])

        if self.arc_num == 1:
            x = self.layer1(rx[:,-1])
            x = self.bn1(x)
            x = self.layer2(x)
            x = self.bn2(x)
            output = self.layer3(x)

        if self.arc_num in [2,3]:
            x = self.layer1(rx[:,-1])
            x = self.bn1(x)
            x = self.layer2(x)
            x = self.bn2(x)
            x = self.layer3(x)
            x = self.bn3(x)
            output = self.layer4(x)

        output if len(output.shape) > 1 else output.unsqueeze(0)
        return output





def get_rnn_model(
    col_feature, train_loader, val_loader,
    epochs, batch_size, learning_rate, window_size, hidden_size,
    use_early_stop=False, use_weighted_sampler=False, class_weights=None,
    use_dual_lstm=False, use_gru_model=False,
):

    input_size = len(col_feature)
    output_size = 3
    #use_early_stop = False

    if use_weighted_sampler:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # encoder = RNNEncoder(input_size, hidden_size, device).to(device)
    # decoder = RNNDecoder(hidden_size, output_size).to(device)
    # model = RNNSeq2Seq(encoder, decoder).to(device)

    if use_gru_model:
        model = GRUmodel(input_size, hidden_size, output_size).to(device)
    else:
        model = LSTMmodel(input_size, hidden_size, output_size, use_dual_lstm=use_dual_lstm).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    if use_early_stop:
        early_stopping = EarlyStopping(tolerance=5, min_delta=0.01)

    # print("Begin training.")
    for e in range(1, epochs+1):
        # TRAINING
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

        # VALIDATION
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

        if use_early_stop:
            early_stopping(train_epoch_loss/len(train_loader), val_epoch_loss/len(val_loader))
        # print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

        if use_early_stop and early_stopping.early_stop:
            break

    return model



def get_predictions(test_loader, model):
    y_pred_list = []
    y_score_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_score, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_score_list.append(y_pred_score.cpu().numpy())
            y_pred_list.append(y_pred_tags.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_score_list = [a.squeeze().tolist() for a in y_score_list]
    return (y_pred_list, y_score_list)




#epochs=both_epochs; window_size=both_window_size; hidden_size=both_hidden_size; use_dual_lstm=both_use_dual_lstm; use_gru_model=both_use_gru_model
def run_mlp_pipeline(col_feature, col_target, train_df, valid_df, test_df, 
                     epochs, batch_size, learning_rate, window_size, hidden_size, 
                     use_early_stop=False, use_weighted_sampler=False,
                     use_dual_lstm=False, use_gru_model=False):
    set_seed(100)
    (train_loader, val_loader, test_loader, weighted_sampler, class_weights) = get_torch_rnn_dataloaders(col_feature, col_target, train_df, valid_df, test_df, window_size, batch_size,
        use_weighted_sampler=use_weighted_sampler
    )
    model = get_rnn_model(col_feature, train_loader, val_loader,epochs, batch_size, learning_rate, window_size, hidden_size, 
        use_early_stop=use_early_stop, use_weighted_sampler=use_weighted_sampler, class_weights=class_weights,
        use_dual_lstm=use_dual_lstm, use_gru_model=use_gru_model
    )
    (y_pred_list, y_score_list) = get_predictions(test_loader, model)

    return (y_pred_list, y_score_list)




##############################################################################
## Settings

config = load_config(path=configPath / "settings.yml")

# col_date = ['datetime','symbolTime']
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

year_vec = np.array(sorted(list(set(data_df['year']))))
price_vec = data_df["close_D1"].values




save_vmd_feat_dict = False
vmd_alpha = 2000
vmd_lookback = 120
vmd_k = 4

if save_vmd_feat_dict:
    feat_dict_list = []
    for i in range(1, len(price_vec)+1):
        if (i % 100) == 0:
            print(f"{i}: {len(price_vec)}")

        if i < vmd_lookback:
           feat_dict = {}
           for j in range(vmd_k):
               feat_dict[f"VMD_{j}"] = np.nan

        else:
            tsv = price_vec[(i-vmd_lookback):i]
            tso = VMD(tsv, vmd_k, alpha=vmd_alpha, tau=0, DC=False, init=1, tol=1e-7)
            feat_dict = {}
            for j in range(vmd_k):
               feat_dict[f"VMD_{j}"] = tso[-1,j]

        feat_dict_list += [deepcopy(feat_dict)]

    vmd_feat_df = pd.DataFrame(feat_dict_list)
    vmd_feat_dict = {}
    vmd_feat_dict["data"] = vmd_feat_df.copy()
    save_obj(vmd_feat_dict, "vmd_feat_dict")
else:
    vmd_feat_dict = load_obj("vmd_feat_dict")
    vmd_feat_df = vmd_feat_dict["data"].copy()
    data_df = pd.concat([data_df, vmd_feat_df], axis=1)








vg_k = 3
# centrality_type_list = ['degree']
# # 120: 0.531, 43 / 758
# # 60: 0.550, 61 / 756
# 
# centrality_type_list = ['closeness']
# # 120: 0.570, 57 / 758
# # 60: 0.581, 36 / 756
# 
# centrality_type_list = ['harmonic']
# # 120: 0.609, 28 / 758
# # 60: 0.616, 61 / 756



save_vg_feat_dict = False
vg_lookback_list = [60, 120]
centrality_type_map_dict = {
    'degree': nx.degree_centrality,
    'closeness': nx.closeness_centrality,
    # 'eigenvector': nx.eigenvector_centrality,
    'harmonic': nx.harmonic_centrality,
}
vg_max_k = 3

if save_vg_feat_dict: 
    for vg_lookback in vg_lookback_list:
        feat_dict_list = []
        for i in range(1, len(price_vec)+1):
            if (i % 100) == 0:
                print(f"{i}: {len(price_vec)}")
    
            if i < vg_lookback:
                feat_dict = {}
                for vg_k in range(vg_max_k):
                    for centrality_type in centrality_type_map_dict:
                        feat_dict[f"VG_{centrality_type}_{vg_lookback}_{vg_k}"] = np.nan
            else:
                tsv = price_vec[(i-vg_lookback):i]
                tsg = visibility_graph(tsv)
                feat_dict = {}
                for vg_k in range(vg_max_k):
                    tsg_shell = nx.k_shell(tsg, vg_k+1)
                    for centrality_type in centrality_type_map_dict:
                        centralities = list(centrality_type_map_dict[centrality_type](tsg_shell).values())
                        feat_dict[f"VG_{centrality_type}_{vg_lookback}_{vg_k}"] = np.mean(centralities)
    
            feat_dict_list += [deepcopy(feat_dict)]
    
        vg_feat_df = pd.DataFrame(feat_dict_list)
        vg_feat_dict = {}
        vg_feat_dict["data"] = vg_feat_df.copy()
        save_obj(vg_feat_dict, f"vg_feat_dict_{vg_lookback}")
else:
    for vg_lookback in vg_lookback_list:
        vg_feat_dict = load_obj(f"vg_feat_dict_{vg_lookback}")
        vg_feat_df = vg_feat_dict["data"].copy()
        data_df = pd.concat([data_df, vg_feat_df], axis=1)




cols = []
for vg_lookback in vg_lookback_list:
    for vg_k in range(vg_max_k):
        for centrality_type in centrality_type_map_dict:
            cols += [f"VG_{centrality_type}_{vg_lookback}_{vg_k}"]


##-##
# data_df.drop(['VG_degree_0','VG_closeness_0','VG_harmonic_0'], axis=1, inplace=True)

# data_df['VG_degree_0'] = data_df['VG_degree_0'].fillna(1)
# data_df['VG_closeness_0'] = data_df['VG_closeness_0'].fillna(0)
# data_df['VG_harmonic_0'] = data_df['VG_harmonic_0'].fillna(0)

# data_df['VG_degree_0'] = data_df['VG_degree_0'].fillna(-1)
# data_df['VG_closeness_0'] = data_df['VG_closeness_0'].fillna(-1)
# data_df['VG_harmonic_0'] = data_df['VG_harmonic_0'].fillna(-1)



# data_df['VG_degree_0'].value_counts()
# 
# 
# data_df['VG_closeness_0'].value_counts()
# data_df['VG_harmonic_0'].value_counts()
# 
# 
# 
# data_df[['VG_degree_0','VG_closeness_0','VG_harmonic_0']].median()


##-##



# feature_type_list = ['VMD','VG_degree_60','VG_closeness_120','VG_harmonic_60']

# feature_type_list = ['VMD']
# 
# # combi: 0.576, 99 / 1,476
# # combi: 0.566, 56 / 1,476
# # combi: 0.585, 62 / 1,476
# # combi: 0.600, 39 / 1,476
# 
# feature_type_list = ['VMD','VG_harmonic_60']
# # combi: 0.588, 114 / 1,476 **
# 
# feature_type_list = ['VMD','VG_closeness_120']
# # combi: 0.582, 138 / 1,476
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_closeness_120']
# # combi: 0.584, 146 / 1,476
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_closeness_120','VG_degree_60']
# # combi: 0.582, 163 / 1,476
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_harmonic_120','VG_closeness_120','VG_degree_60']
# # combi: 0.588, 167 / 1,476
# 
# 
# feature_type_list = ['VMD','VG_harmonic_60']
# # combi: 0.588, 114 / 1,476 **
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_closeness_60']
# # combi: 0.580, 119 / 1,476 **
# 
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_degree_60']
# # combi: 0.595, 132 / 1,476 **
# 
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_degree_120']
# # combi: 0.562, 132 / 1,476
# 
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_harmonic_120']
# # combi: 0.603, 129 / 1,476 **
# 
# 
# feature_type_list = ['VMD','VG_harmonic_120']
# # combi: 0.597, 120 / 1,476 **
# 
# 
# 
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_degree_60']
# # combi: 0.595, 132 / 1,476
# 
# 
# feature_type_list = ['VMD','VG_harmonic_120','VG_degree_60']
# # combi: 0.609, 137 / 1,476
# 
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_harmonic_120','VG_degree_60']
# # combi: 0.594, 149 / 1,476
# 
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_harmonic_120']
# # combi: 0.603, 129 / 1,476
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_harmonic_120','VG_closeness_60']
# # combi: 0.592, 135 / 1,476
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_harmonic_120','VG_closeness_120']
# # combi: 0.587, 152 / 1,476
# 
# 
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_harmonic_120','VG_closeness_60']
# # combi: 0.592, 135 / 1,476
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_harmonic_120','VG_closeness_60','VG_degree_60']
# # combi: 0.585, 152 / 1,476
# 
# feature_type_list = ['VMD','VG_harmonic_60','VG_harmonic_120','VG_closeness_60','VG_degree_120']
# # combi: 0.569, 149 / 1,476


##-##
# feature_type_list = ['VMD','VG_harmonic_60','VG_harmonic_120','VG_closeness_60','VG_degree_60']
feature_type_list = ['VMD']
##-##

# combi: 0.585, 152 / 1,476

# feature_type_list = ['VMD','VG_harmonic_60','VG_harmonic_120','VG_closeness_60','VG_degree_60','TECH']
# # combi: 0.563, 165 / 1,476


##-##

# col_feature_both = [f"VMD_{j}" for j in range(vmd_k)] + [
#     # 'D1-SMA10-val',
#     # 'D1-MACD-macd',
#     # 'D1-MACD-macdsignal',
#     # 'D1-MACD-macdhist',
#     # 'D1-BB20-upper',
#     # 'D1-BB20-mid',
#     # 'D1-BB20-lower',
#     # 'D1-ROC2-val',
#     # 'D1-MOM4-val',
#     # 'D1-RSI10-val',
#     # 'D1-CCI20-val',
# ]



# centrality_type = ['degree','closeness','harmonic'][0]

# col_feature_both = []
# for k in range(vg_k):
#     for centrality_type in centrality_type_list:
#         col_feature_both += [f"VG_{centrality_type}_{k}"]



# baseline: 0.576, 99 / 1,475
# both: 0.580, 87 / 1,475
##-##





## General GRU Settings (With T2V)
both_use_gru_model = True
both_use_dual_lstm = False
both_epochs = 1
both_hidden_size = 50
both_window_size = 5

batch_size = 8
thres_multiplier = 3
use_early_stop = False
learning_rate = 0.0005


use_weighted_sampler = False
volatility_type = 'thres_auto_v1'
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

    if volatility_type == 'thres_auto_v1':
        thres_df = data_df_temp[data_df_temp['year_month'].isin(train_year_month_list)].copy()
        thres_df.reset_index(drop=True, inplace=True)
        col_target_gains_thres = get_threshold(thres_df[col_price], stepsize=threshold_stepsize) * thres_multiplier
        data_df_temp[col_target] = 1
        data_df_temp.loc[data_df_temp[col_target_gains] < -col_target_gains_thres, col_target] = 0
        data_df_temp.loc[data_df_temp[col_target_gains] > col_target_gains_thres, col_target] = 2


    pred_df = data_df_temp[data_df_temp['year_month'].isin(test_year_month_list)].copy()
    pred_df.reset_index(drop=True, inplace=True)

    for feature_type in feature_type_list:
        if feature_type == "VMD":
            col_feature = [col_price] + [f"VMD_{j}" for j in range(vmd_k)]

        if feature_type.split('_')[0] == "VG":
            centrality_type = feature_type.split('_')[1]
            vg_lookback = feature_type.split('_')[2]
            col_feature = [col_price]
            for k in range(vg_k):
                col_feature += [f"VG_{centrality_type}_{vg_lookback}_{k}"]

        if feature_type == "TECH":
            col_feature = [col_price] + [
                'D1-SMA10-val',
                'D1-MACD-macd',
                'D1-MACD-macdsignal',
                'D1-MACD-macdhist',
                'D1-BB20-upper',
                'D1-BB20-mid',
                'D1-BB20-lower',
                'D1-ROC2-val',
                'D1-MOM4-val',
                'D1-RSI10-val',
                'D1-CCI20-val',
                # 'D1-PSAR-val',
            ]

        model_df = data_df_temp[['datetime','year_month',col_target]+col_feature].copy()
        model_df.reset_index(drop=True, inplace=True)

        model_df = model_df.dropna()
        model_df.reset_index(drop=True, inplace=True)

        train_df = model_df[model_df['year_month'].isin(train_year_month_list)].copy()
        valid_df = model_df[model_df['year_month'].isin(valid_year_month_list)].copy()
        test_df = model_df[model_df['year_month'].isin(test_year_month_list)].copy()

        valid_df_windowed = pd.concat([train_df,valid_df]).copy()
        valid_df_windowed = valid_df_windowed.tail(len(valid_df) + both_window_size-1)

        test_df_windowed = pd.concat([train_df, valid_df, test_df]).copy()
        test_df_windowed = test_df_windowed.tail(len(test_df) + both_window_size-1)

        (y_pred_list, y_score_list) = run_mlp_pipeline(
            col_feature, col_target, train_df, valid_df_windowed, test_df_windowed,
            both_epochs, batch_size, learning_rate, both_window_size, both_hidden_size,
            use_early_stop=use_early_stop, use_weighted_sampler=use_weighted_sampler,
            use_dual_lstm=both_use_dual_lstm, use_gru_model=both_use_gru_model,
            )
        test_df[f'pred_{feature_type}'] = y_pred_list
        test_df[f'score_{feature_type}'] = y_score_list

        test_df.reset_index(drop=True, inplace=True)
        merge_df = pd.merge(pred_df['datetime'], test_df, how='left', on='datetime')
        pred_df[f'pred_{feature_type}'] = merge_df[f'pred_{feature_type}'].values
        pred_df[f'pred_{feature_type}'] = pred_df[f'pred_{feature_type}'].fillna(1)


    def get_pred_combi(row, feature_type_list):
        pred_list = [(row[f'pred_{feature_type}']-1) for feature_type in feature_type_list]
        pred_list = [x for x in pred_list if x != 0]
        if len(pred_list) == 0:
            out = 1
        else:
            out = int(np.sign(np.mean(pred_list))) + 1
        return out

    pred_df['pred_combi'] = pred_df.apply(get_pred_combi, axis=1, feature_type_list=feature_type_list)

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

##_##
plt_title = f"{pred_type}: {profit_accuracy:,.3f}, {pred_count:,.0f} / {total_count:,.0f}"
# plt_title = f"vmd_k {vmd_k}: {profit_accuracy:,.3f}, {pred_count:,.0f} / {total_count:,.0f}"
##-##

plt.title(plt_title)

for test_year in [2018, 2019, 2020, 2021, 2022, 2023]:
    col_pred = f'pred_{pred_type}'
    (profit_accuracy, pred_count,
     total_count,
     short_accuracy, long_accuracy) = get_profit_accuracy(plot_df[plot_df['year'] == test_year], col_pred, col_target_gains)
    plt.text(datetime(test_year,1,1), 10, f"{profit_accuracy:,.3f}")
plt.show()
print(f"# {plt_title}")









