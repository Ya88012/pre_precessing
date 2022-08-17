import pandas as pd
import torch
import time
import numpy as np

def MAPELoss(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

start_time = time.perf_counter()

df_train = np.load('npy_files/training_scalar_volume_array.npy')
df_test = np.load('npy_files/testing_scalar_volume_array.npy')

print('df_train.shape:')
print(df_train.shape)
print('df_test.shape:')
print(df_test.shape)

threshold = 10

total_MSE_departing_value = 0
total_MSE_arriving_value = 0
total_MAPE_departing_value = 0
total_MAPE_arriving_value = 0
total_depart_counter = 0
total_arriving_counter = 0

# 2020 / 7 / 1 ===> Wed
training_weekday_start_time_slot = [48 * 5, 48 * 6, 0, 48 * 1, 48 * 2, 48 * 3, 48 * 4]
# 2020 / 8 / 10 ===> Mon
testing_weekday_start_time_slot = [0, 48 * 1, 48 * 2, 48 * 3, 48 * 4, 48 * 5, 48 * 6]

for major_flow_station_index in range(100):
    for day in range(0, 7):
        for hour in range(18, 37):

            # print('major_flow_station_index:', major_flow_station_index)
            # print('day:', day)
            # print('hour:', hour)

            train_arriving_flow_list = df_train[ training_weekday_start_time_slot[day] + hour :: 48 * 7, major_flow_station_index, 0]
            train_departing_flow_list = df_train[ training_weekday_start_time_slot[day] + hour :: 48 * 7, major_flow_station_index, 1]
            
            test_arriving_flow_list = df_test[ testing_weekday_start_time_slot[day] + hour :: 48 * 7, major_flow_station_index, 0]
            test_departing_flow_list = df_test[ testing_weekday_start_time_slot[day] + hour :: 48 * 7, major_flow_station_index, 1]
            
            # print( 'train_arriving_flow_list:', train_arriving_flow_list )
            # print( 'test_arriving_flow_list:', test_arriving_flow_list )
            # print( 'train_departing_flow_list:', train_departing_flow_list )
            # print( 'test_departing_flow_list:', test_departing_flow_list )

            loss_func = torch.nn.MSELoss()

            for a_value in test_arriving_flow_list:
                if a_value < threshold:
                    continue
                MSE_arriving = loss_func(torch.tensor(train_arriving_flow_list.mean()), torch.tensor(a_value)).item()
                MAPE_arriving = MAPELoss(train_arriving_flow_list.mean(), a_value)
                total_MSE_arriving_value += MSE_arriving
                total_MAPE_arriving_value += MAPE_arriving
                # print('MSE_arriving:', MSE_arriving)
                # print("MAPE_arriving:", MAPE_arriving)
                total_arriving_counter += 1

            for d_value in test_departing_flow_list:
                if d_value < threshold:
                    continue
                MSE_departing = loss_func(torch.tensor(train_departing_flow_list.mean()), torch.tensor(d_value)).item()
                MAPE_departing = MAPELoss(train_departing_flow_list.mean(), d_value)
                total_MSE_departing_value += MSE_departing
                total_MAPE_departing_value += MAPE_departing
                # print('MSE_departing:', MSE_departing)
                # print("MAPE_departing:", MAPE_departing)
                total_depart_counter += 1
      
            # print()

print('total_MSE_arriving_value:', total_MSE_arriving_value)
print('total_MSE_departing_value:', total_MSE_departing_value)

print('total_MAPE_arriving_value:', total_MAPE_arriving_value)
print('total_MAPE_departing_value:', total_MAPE_departing_value)

print('total_arriving_counter:', total_arriving_counter)
print('total_depart_counter:', total_depart_counter)

RMSE_arriving = (total_MSE_arriving_value / total_arriving_counter) ** 0.5
RMSE_departing = (total_MSE_departing_value / total_depart_counter) ** 0.5

MAPE_arriving = total_MAPE_arriving_value / total_arriving_counter
MAPE_departing = total_MAPE_departing_value / total_depart_counter

print('RMSE_arriving:', RMSE_arriving)
print('RMSE_departing:', RMSE_departing)

print("MAPE_arriving:", MAPE_arriving)
print("MAPE_departing:", MAPE_departing)

end_time = time.perf_counter()

print("used_time:", end_time - start_time)
