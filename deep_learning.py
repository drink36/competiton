import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
# Import necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle, resample
from math import sqrt

# Keras specific
import keras
from keras import layers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout


def read_data(file_name):
    data = pd.read_csv(file_name)
    return data


def resample_data(data):
    majority = data[data.Attribute17 == 0]
    minority = data[data.Attribute17 == 1]
    # temp_majority = resample(majority, replace=False,
    #                          n_samples=500, random_state=123)
    # temp_minority = resample(minority, replace=False,
    #                          n_samples=500, random_state=123)
    # validation_data = pd.concat([temp_minority, temp_majority])
    # majority = majority.drop(temp_majority.index)
    # minority = minority.drop(temp_minority.index)
    print("majority", majority.shape)
    print("minority", minority.shape)
    # Downsample majority class
    majority_down_sampled = resample(
        majority, replace=False, n_samples=5000, random_state=7414)
    minority_up_sampled = resample(
        minority, replace=True, n_samples=5000, random_state=7414)
    data = pd.concat([majority_down_sampled, minority_up_sampled])
    data = shuffle(data)
    return data


def model(file_name, epoch_num, batch_size_num):
    t_df = pd.read_csv(file_name)
    target_column = ['sar_flag']
    predictors = list(set(list(t_df.columns)) - set(target_column))
    v_df = pd.read_csv("data_after_process/test.csv")
    y_train = t_df[target_column].values
    scaler = StandardScaler()
    t_df[predictors] = scaler.fit_transform(t_df[predictors])
    v_df[predictors] = scaler.transform(v_df[predictors])
    x_train = t_df[predictors].values
    x_test = v_df[predictors].values
    # one hot encode outputs
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    model = Sequential()
    # Every layer has one input and output tensor
    # The first layer needs to know the input shape
    model.add(Dense(300, input_dim=9, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    # The main advantage of the "adam" optimizer is that we don't need to specify the learning rate.

    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"],)
    history = model.fit(x_train, y_train, validation_split=0.33, epochs=epoch_num,
                        batch_size=batch_size_num, verbose=1)
    pred_train = model.predict(x_test)
    result = model.evaluate(x_train, y_train)
    return result, pred_train, history


path = "data_after_process/train.csv"
out = read_data('data_after_process/test.csv')

result = model(path, 10, 30)
output = pd.DataFrame(result[1])
history = pd.DataFrame(result[2].history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
output = output.round(7)
output = output.sort_values(by=[0], ascending=False)
output.index = out['alert_key']
output.index.name = 'alert_key'
output.columns = ['probability']
print(output)
print(result[0])
output.to_csv("submit.csv", header=['probability'], index_label='alert_key')
sample = read_data('first/predict.csv')
out = read_data('submit.csv')
sample = sample.merge(out, on='alert_key', how='left')

sample = sample[sample['probability_y'].isnull()]
print(sample)
sample.rename(columns={'probability_x': 'probability'}, inplace=True)
sample.drop(['probability_y'], axis=1, inplace=True)
out = out.append(sample, ignore_index=True)
out.to_csv("submit.csv", index=False)
# for i in range(len(sample)):
#     if(sample['alert_key'][i] in out['alert_key']):
#         sample.drop(i)
#         i -= 1
#out = out.append(sample, ignore_index=True)
# print(out)
# out.to_csv("submit.csv")
# f = open(path, 'w')
# for name, value in result:
#     print(name, ': ', value)
#     f.write(name+': '+str(value)+'\n')
# f.close()
