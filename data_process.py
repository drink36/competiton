import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from data_config import (DataSource, FeatureType,
                         CCBAConfig, CDTXConfig, DPConfig, REMITConfig, CUSTINFOConfig,
                         CONFIG_MAP)
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from tqdm import tqdm
import seaborn as sns
TRAIN_PATH = 'first'
CCBA_PATH = 'first/public_train_x_ccba_full_hashed.csv'
CDTX_PATH = 'first/public_train_x_cdtx0001_full_hashed.csv'
CUSTINFO_PATH = 'first/public_train_x_custinfo_full_hashed.csv'
DP_PATH = 'first/public_train_x_dp_full_hashed.csv'
REMIT_PATH = 'first/public_train_x_remit1_full_hashed.csv'
PDATE_PATH = 'first/public_x_alert_date.csv'
TDATE_PATH = 'first/train_x_alert_date.csv'
ANSWER_PATH = 'first/train_y_answer.csv'
SAMPLE_PATH = './sample_submission.csv'
ccba = pd.read_csv(CCBA_PATH)
cdtx = pd.read_csv(CDTX_PATH)
custinfo = pd.read_csv(CUSTINFO_PATH)
dp = pd.read_csv(DP_PATH)
remit = pd.read_csv(REMIT_PATH)
pdate = pd.read_csv(PDATE_PATH)
tdate = pd.read_csv(TDATE_PATH)
answer = pd.read_csv(ANSWER_PATH)

names = ['ccba', 'cdtx', 'custinfo', 'dp',
         'remit', 'pdate', 'tdate', 'answer']
datas = [ccba, cdtx, custinfo, dp, remit, pdate, tdate, answer]


def pic(data):
    corr = data.corr()

    plt.figure(figsize=(20, 20))
    g = sns.heatmap(data=corr,  vmin=corr.values.min(), vmax=1, square=True, cmap="YlGnBu", linewidths=0.1, annot=True, xticklabels=1, yticklabels=1  # 矩阵数据集，数据的index和columns分别为heatmap的y轴方向和x轴方向标签
                    )
    g.set_xticklabels(g.get_xticklabels(), rotation=45,
                      horizontalalignment='right')
    g.set_yticklabels(g.get_yticklabels(), rotation=45)
    plt.show()


date = pd.concat([pdate, tdate], axis=0)
custinfo = custinfo.merge(date, on='alert_key', how='left')
custinfo = custinfo.merge(answer, on='alert_key', how='left')
new_temp = dp
new_temp['tx_amt'] = new_temp['tx_amt']*new_temp['exchg_rate']
new_temp.drop(
    ['exchg_rate', 'fiscTxId', 'txbranch', 'tx_type', 'info_asset_code', 'cross_bank', 'ATM'], axis=1, inplace=True)
new_temp['tx_time'] = new_temp['tx_time'].apply(lambda x: 1)
new_temp = new_temp.groupby(
    by=['cust_id', 'tx_date']).sum().reset_index()
new_temp.rename(columns={'tx_date': 'date'}, inplace=True)
new_temp.rename(columns={'tx_amt': 'amt'}, inplace=True)
new_temp2 = cdtx
new_temp2.drop(['country', 'cur_type'], axis=1, inplace=True)
new_temp2 = new_temp2.groupby(by=['cust_id', 'date']).sum().reset_index()
# new_temp.to_csv('data_after_process/dp.csv', index=False)
# new_temp2.to_csv('data_after_process/cdtx.csv', index=False)
# pro = ProfileReport(new_temp)
# pro2 = ProfileReport(new_temp2)
# pro.to_file('dp.html')
# pro2.to_file('cdtx.html')
custinfo = custinfo.merge(new_temp, on=['cust_id', 'date'], how='left')
custinfo = custinfo.merge(new_temp2, on=['cust_id', 'date'], how='left')
custinfo['amt_y'] = custinfo['amt_y'].fillna(0)
custinfo['amt_x'] = custinfo['amt_x'].fillna(0)
custinfo['amt_x'] = custinfo['amt_x'] + custinfo['amt_y']
custinfo.drop(['amt_y'], axis=1, inplace=True)
custinfo.rename(columns={'amt_x': 'amt'}, inplace=True)
# test fillna(0)
le = LabelEncoder()
custinfo['cust_id'] = le.fit_transform(custinfo['cust_id'])
df_sar = custinfo['sar_flag']
custinfo.drop(['sar_flag'], axis=1, inplace=True)
custinfo.insert(len(custinfo.columns), 'sar_flag', df_sar)
test = custinfo[custinfo['sar_flag'].isnull()]
test = test.drop(['sar_flag'], axis=1)
train = custinfo[custinfo['sar_flag'].notnull()]
custinfo.fillna(0, inplace=True)
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
pic(train)
test.to_csv('data_after_process/test.csv', index=False)
train.to_csv('data_after_process/train.csv', index=False)
pro3 = ProfileReport(custinfo)
pro3.to_file('custinfo.html')
custinfo.to_csv('data_after_process/custinfo_out.csv', index=False)
