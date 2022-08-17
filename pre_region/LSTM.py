from random import sample
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable as V
import torch.optim as optim
import os

# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, batch_size, batch_first):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.batch_first = batch_first

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = batch_first)

    def forward(self, input_data, h0 = None, c0 = None):
        if h0 != None and c0 != None:
            output, (hn, cn) = self.lstm(input_data, (h0.cuda(), c0.cuda()))
        else:
            output, (hn, cn) = self.lstm(input_data)
        return output, (hn, cn)

training_vdata = np.load('npy_files/training_vdata_array.npy')
testing_vdata = np.load('npy_files/testing_vdata_array.npy')

w = 10
h = 20

def file_loader(data_array):

    week_day = 3
    day_time_slot_num = 48
    time_start = week_day * day_time_slot_num
    time_end = data_array.shape[0]
    lstm_feature = []
    all_label_feature = []

    for x in range(w):
        for y in range(h):
            region_feature = []
            label_feature = []

            for t in range(time_start, time_end):
                real_t = t - ( week_day * day_time_slot_num )
                # week_feature is the same time slot in a day for past a week
                week_feature = data_array[real_t : t : day_time_slot_num, x, y, :]
                # yesterday_feature is the all time slot for yesterday
                yesterday_feature = data_array[t - day_time_slot_num : t, x, y, :]
                # print(week_feature.shape)
                # print(yesterday_feature.shape)
                all_need_feature = np.concatenate(week_feature, yesterday_feature))
                # all_need_feature = week_feature
                # all_need_feature = yesterday_feature
                # print(all_need_feature.shape)
                region_feature.append(all_need_feature)
                label_feature.append(data_array[t, x, y, :])

            lstm_feature.append(region_feature)
            all_label_feature.append(label_feature)

    feture_data = np.array(lstm_feature, dtype = 'int32')
    print(feture_data)
    print(feture_data.shape)
    label_data = np.array(all_label_feature, dtype = 'int32')
    print(label_data)
    print(label_data.shape)
    return feture_data, label_data

def eval_lstm(y, pred_y, threshold):
    
    print('y:', y)
    print('y.shape:', y.shape)
    print('pred_y.shape:', pred_y.shape)
    print('pred_y:', pred_y)
    print('y.max():', y.max())
    print('y.min():', y.min())
    print('pred_y.max():', pred_y.max())
    print('pred_y.min():', pred_y.min())

    pickup_y = y[:, 0]
    dropoff_y = y[:, 1]
    pickup_pred_y = pred_y[:, 0]
    dropoff_pred_y = pred_y[:, 1]
    pickup_mask = pickup_y > threshold
    dropoff_mask = dropoff_y > threshold

    avg_pickup_mape = -1
    avg_pickup_rmse = -1
    avg_dropoff_mape = -1
    avg_dropoff_rmse  = -1

    print('pickup_y[pickup_mask]:', pickup_y[pickup_mask])
    print('pickup_pred_y[pickup_mask]:', pickup_pred_y[pickup_mask])
    print('dropoff_y[dropoff_mask]:', dropoff_y[dropoff_mask])
    print('dropoff_pred_y[dropoff_mask]:', dropoff_pred_y[dropoff_mask])

    # pickup part
    if np.sum(pickup_mask) != 0:
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask]) / pickup_y[pickup_mask])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask])))
    # dropoff part
    if np.sum(dropoff_mask) != 0:
        avg_dropoff_mape = np.mean(np.abs(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask]) / dropoff_y[dropoff_mask])
        avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask])))

    return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)

train_sample_data, train_label_data = file_loader(training_vdata)
test_sample_data, test_label_data = file_loader(testing_vdata)

print(train_sample_data.shape)
print(train_label_data.shape)
print(test_sample_data.shape)
print(test_label_data.shape)

# for x in range(w):
#     for y in range(h):

#         region = x * h + y
#         print('region:', region)
#         print('train_label_data[region, :, 0].max():', train_label_data[region, :, 0].max())
#         print('train_label_data[region, :, 1].max():', train_label_data[region, :, 1].max())

#         print('test_label_data[region, :, 0].max():', test_label_data[region, :, 0].max())
#         print('test_label_data[region, :, 1].max():', test_label_data[region, :, 1].max())

input_size = 2
hidden_size = 2
num_layers = 1
batch_size = 1
batch_first = True
epochs = 30
threshold = 15

pickup_counter = 0
t_avg_pickup_rmse = 0
t_avg_pickup_mape = 0

dropoff_counter = 0
t_avg_dropoff_rmse = 0
t_avg_dropoff_mape = 0

for region in range( 200 ):

    lstm = LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_size = batch_size, batch_first = batch_first)
    optimizer = optim.Adam(lstm.parameters(), lr = 0.01)
    loss_fnc = nn.MSELoss()

    # normalize
    train_max = train_sample_data.max() if train_sample_data.max() > train_label_data.max() else train_label_data.max()
    train_min = train_sample_data.min() if train_sample_data.min() < train_label_data.min() else train_label_data.min()
    test_max = test_sample_data.max() if test_sample_data.max() > test_label_data.max() else test_label_data.max()
    test_min = test_sample_data.min() if test_sample_data.min() < test_label_data.min() else test_label_data.min()

    n_max = train_max if train_max > test_max else test_max
    n_min = train_min if train_min < test_min else test_min

    print('region:', region)

    test_label_mask = test_label_data[region] > threshold
    if np.sum(test_label_mask) == 0:
        print('less than threshold!')
        continue
    else:
        print('np.sum(test_label_mask):', np.sum(test_label_mask))

    # training

    input_data_array = train_sample_data[region]
    label_data_array = train_label_data[region]

    if torch.cuda.is_available():
        print('torch.cuda.is_available():', torch.cuda.is_available())
        lstm.cuda()

    train_batch_num = int(train_sample_data.shape[1] / batch_size)

    for i in range(epochs):

        training_loss = 0
        print('epoch:', i)

        for j in range( train_batch_num ):

            input_data = torch.Tensor( (input_data_array[j, :, :].reshape(-1, 48, 2) - n_min) / (n_max - n_min) )
            label_data = torch.Tensor( (label_data_array[j, :].reshape(1, 1, 2) - n_min) / (n_max - n_min) )

            if torch.cuda.is_available():
                input_data = input_data.cuda()
                label_data = label_data.cuda()

            # print('input_data.shape:', input_data.shape)
            # print('label_data.shape:', label_data.shape)

            optimizer.zero_grad()

            # h0 = torch.randn(num_layers, batch_size, hidden_size)
            # c0 = torch.randn(num_layers, batch_size, hidden_size)

            output, (hn, cn) = lstm(input_data)

            # print('output:', output)
            # print('output.shape:', output.shape)
            # print('hn:', hn)
            # print('hn.shape:', hn.shape)

            loss = loss_fnc(hn, label_data)

            training_loss += loss.item()

            loss.backward()
            optimizer.step()

        print('training_loss:', training_loss)

    # testing

    input_data_array = test_sample_data[region]
    label_data_array = test_label_data[region]

    test_batch_num = int(test_sample_data.shape[1] / batch_size)

    # lstm.eval()
    total_loss = 0

    input_data = torch.Tensor( (input_data_array - n_min) / (n_max - n_min) )

    if torch.cuda.is_available():
        input_data = input_data.cuda()

    print('input_data.shape:', input_data.shape)

    # h0 = torch.randn(num_layers, 624, hidden_size)
    # c0 = torch.randn(num_layers, 624, hidden_size)

    output, (hn, cn) = lstm(input_data)

    print('hn:', hn)
    print('hn.shape:', hn.shape)

    (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape) = eval_lstm( label_data_array.reshape(-1, 2), hn.cpu().detach().numpy().reshape(-1, 2) * (n_max - n_min) + n_min, threshold )

    if avg_pickup_rmse != -1 and avg_pickup_mape != -1:
        t_avg_pickup_rmse += avg_pickup_rmse * np.sum(test_label_mask[:, 0])
        t_avg_pickup_mape += avg_pickup_mape * np.sum(test_label_mask[:, 0])
        pickup_counter += np.sum(test_label_mask[:, 0])

    if avg_dropoff_rmse != -1 and avg_dropoff_mape != -1:
        t_avg_dropoff_rmse += avg_dropoff_rmse * np.sum(test_label_mask[:, 1])
        t_avg_dropoff_mape += avg_dropoff_mape * np.sum(test_label_mask[:, 1])
        dropoff_counter += np.sum(test_label_mask[:, 1])

    print('pickup_counter:', pickup_counter)
    print('t_avg_pickup_rmse:', t_avg_pickup_rmse)
    print('t_avg_pickup_mape:', t_avg_pickup_mape)

    print('dropoff_counter:', dropoff_counter)
    print('t_avg_dropoff_rmse:', t_avg_dropoff_rmse)
    print('t_avg_dropoff_mape:', t_avg_dropoff_mape)

print('pickup_counter:', pickup_counter)
print('t_avg_pickup_rmse:', t_avg_pickup_rmse / pickup_counter)
print('t_avg_pickup_mape:', t_avg_pickup_mape / pickup_counter)

print('dropoff_counter:', dropoff_counter)
print('t_avg_dropoff_rmse:', t_avg_dropoff_rmse / dropoff_counter)
print('t_avg_dropoff_mape:', t_avg_dropoff_mape / dropoff_counter)
