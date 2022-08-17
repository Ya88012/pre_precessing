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
        if h0 == None and c0 == None:
            output, (hn, cn) = self.lstm(input_data)
        else:
            output, (hn, cn) = self.lstm(input_data, (h0.cuda(), c0.cuda()))
        return output, (hn, cn)

# testing
input_size = 2
hidden_size = 2
num_layers = 1
batch_size = 1
batch_first = True
epochs = 30
threshold = 0
week_day = 1

pickup_counter = 0
t_avg_pickup_rmse = 0
t_avg_pickup_mape = 0

dropoff_counter = 0
t_avg_dropoff_rmse = 0
t_avg_dropoff_mape = 0

def file_loader(data_array):

    day_time_slot_num = 48
    time_start = week_day * day_time_slot_num
    time_end = data_array.shape[0]
    lstm_feature = []
    all_label_feature = []

    for major_flow_station_index in range(100):
        major_flow_station_index_feature = []
        label_feature = []

        for t in range(time_start, time_end):
            real_t = t - ( week_day * day_time_slot_num )
            # week_feature is the same time slot in a day for past a week
            week_feature = data_array[real_t : t : day_time_slot_num, major_flow_station_index, :]
            # yesterday_feature is the all time slot for yesterday
            yesterday_feature = data_array[t - day_time_slot_num : t, major_flow_station_index, :]
            # print(week_feature.shape)
            # print(yesterday_feature.shape)
            all_need_feature = np.concatenate((yesterday_feature, week_feature))
            # all_need_feature = week_feature
            # all_need_feature = yesterday_feature
            # print(all_need_feature.shape)
            major_flow_station_index_feature.append(all_need_feature)
            label_feature.append(data_array[t, major_flow_station_index, :])

        lstm_feature.append(major_flow_station_index_feature)
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

training_vdata = np.load('npy_files/training_scalar_volume_array.npy')
testing_vdata = np.load('npy_files/testing_scalar_volume_array.npy')
train_sample_data, train_label_data = file_loader(training_vdata)
test_sample_data, test_label_data = file_loader(testing_vdata)

for major_flow_station_index in range( 100 ):

    test_label_mask = test_label_data[major_flow_station_index] > threshold
    if np.sum(test_label_mask) == 0:
        print('less than threshold!')
        continue
    else:
        print('np.sum(test_label_mask):', np.sum(test_label_mask))

    # normalize
    train_max = train_sample_data[major_flow_station_index].max() if train_sample_data[major_flow_station_index].max() > train_label_data[major_flow_station_index].max() else train_label_data[major_flow_station_index].max()
    train_min = train_sample_data[major_flow_station_index].min() if train_sample_data[major_flow_station_index].min() < train_label_data[major_flow_station_index].min() else train_label_data[major_flow_station_index].min()
    test_max = test_sample_data[major_flow_station_index].max() if test_sample_data[major_flow_station_index].max() > test_label_data[major_flow_station_index].max() else test_label_data[major_flow_station_index].max()
    test_min = test_sample_data[major_flow_station_index].min() if test_sample_data[major_flow_station_index].min() < test_label_data[major_flow_station_index].min() else test_label_data[major_flow_station_index].min()

    n_max = train_max if train_max > test_max else test_max
    n_min = train_min if train_min < test_min else test_min

    device = torch.device("cuda")
    lstm = LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_size = batch_size, batch_first = batch_first)
    lstm.load_state_dict(torch.load('models_weights_{}\model_wight_{}.'.format(week_day, major_flow_station_index)))
    lstm.to(device)

    input_data_array = test_sample_data[major_flow_station_index]
    label_data_array = test_label_data[major_flow_station_index]

    test_batch_num = int(test_sample_data.shape[1] / batch_size)

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
