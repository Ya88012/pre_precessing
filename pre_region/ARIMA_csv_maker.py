import numpy as np
import pandas as pd

train_array = np.load('npy_files/training_vdata_array.npy')
test_array = np.load('npy_files/testing_vdata_array.npy')

train_time_slot_num = (40) * 48
test_time_slot_num = (20) * 48
w = 10
h = 20

for x in range(w):
    for y in range(h):

        region = x * h + y

        if region == 0:

            temp_train_array = train_array[:, x, y, :]
            # print(temp_train_array)
            
            new_train_temp_df = pd.DataFrame(temp_train_array, columns = ['arriving', 'departing'])
            # print(new_train_temp_df)

            # time_slot_and_region_value_array = np.mgrid[0 : time_slot_num : 1, major_station_num].reshape(2, -1).T
            # print(time_slot_and_region_value_array)

            time_slot_array = np.arange(train_time_slot_num)
            region_array = np.repeat(region, train_time_slot_num)

            # print(time_slot_array.shape)
            # print(region_array.shape)

            weekday_counter = 3

            weekday_counter -= 1
            weekday_array = np.zeros(train_time_slot_num, dtype = 'int32')
            for day_index in range( len(weekday_array) ):
                if day_index > 0 and day_index % 48 == 0:
                    weekday_counter = ( weekday_counter + 1) % 7
                weekday_array[day_index] = weekday_counter
            weekday_array += 1
            # weekday_array = np.repeat(weekday_array, major_station_num, axis = 0)

            time_slot_and_region_value_df = pd.DataFrame({'time_slot': time_slot_array, 'region': region_array})

            new_time_slot_and_region_value_df = time_slot_and_region_value_df.merge( pd.DataFrame(weekday_array, columns = ['weekday']), how = 'inner', left_index = True, right_index = True)
            new_time_slot_and_region_value_df = new_time_slot_and_region_value_df.reindex( columns = ['region', 'time_slot', 'weekday'] )

            # print('new_time_slot_and_region_value_df:')
            # print(new_time_slot_and_region_value_df)

            train_final_df = new_time_slot_and_region_value_df.merge(new_train_temp_df, how = 'inner', left_index = True, right_index = True)
            # print('train_final_df:')
            # print(train_final_df)

        else:

            temp_train_array = train_array[:, x, y, :]
            # print(temp_train_array.shape)
            
            new_train_temp_df = pd.DataFrame(temp_train_array, columns = ['arriving', 'departing'])

            # time_slot_and_region_value_array = np.mgrid[0 : time_slot_num : 1, major_station_num].reshape(2, -1).T
            # print(time_slot_and_region_value_array)

            time_slot_array = np.arange(train_time_slot_num)
            region_array = np.repeat(region, train_time_slot_num)

            # print(time_slot_array.shape)
            # print(region_array.shape)

            weekday_counter = 3

            weekday_counter -= 1
            weekday_array = np.zeros(train_time_slot_num, dtype = 'int32')
            for day_index in range( len(weekday_array) ):
                if day_index > 0 and day_index % 48 == 0:
                    weekday_counter = ( weekday_counter + 1) % 7
                weekday_array[day_index] = weekday_counter
            weekday_array += 1
            # weekday_array = np.repeat(weekday_array, major_station_num, axis = 0)

            time_slot_and_region_value_df = pd.DataFrame({'time_slot': time_slot_array, 'region': region_array})

            new_time_slot_and_region_value_df = time_slot_and_region_value_df.merge( pd.DataFrame(weekday_array, columns = ['weekday']), how = 'inner', left_index = True, right_index = True)
            new_time_slot_and_region_value_df = new_time_slot_and_region_value_df.reindex( columns = ['region', 'time_slot', 'weekday'] )

            # print('new_time_slot_and_region_value_df:')
            # print(new_time_slot_and_region_value_df)

            final_temp_df = new_time_slot_and_region_value_df.merge(new_train_temp_df, how = 'inner', left_index = True, right_index = True)
            # print('final_temp_df:')
            # print(final_temp_df)
            train_final_df = pd.concat([train_final_df, final_temp_df])
            # print('train_final_df:')
            # print(train_final_df)

train_final_df.reset_index(drop = True, inplace = True)
print('train_final_df:')
print(train_final_df)

for x in range(w):
    for y in range(h):

        region = x * h + y

        if region == 0:

            temp_test_array = test_array[:, x, y, :]
            # print(temp_test_array.shape)
            
            new_test_temp_df = pd.DataFrame(temp_test_array, columns = ['arriving', 'departing'])

            # time_slot_and_region_value_array = np.mgrid[0 : time_slot_num : 1, major_station_num].reshape(2, -1).T
            # print(time_slot_and_region_value_array)

            time_slot_array = np.arange(test_time_slot_num)
            region_array = np.repeat(region, test_time_slot_num)

            # print(time_slot_array.shape)
            # print(region_array.shape)

            weekday_counter = 1

            weekday_counter -= 1
            weekday_array = np.zeros(test_time_slot_num, dtype = 'int32')
            for day_index in range( len(weekday_array) ):
                if day_index > 0 and day_index % 48 == 0:
                    weekday_counter = ( weekday_counter + 1) % 7
                weekday_array[day_index] = weekday_counter
            weekday_array += 1
            # weekday_array = np.repeat(weekday_array, major_station_num, axis = 0)

            time_slot_and_region_value_df = pd.DataFrame({'time_slot': time_slot_array, 'region': region_array})

            new_time_slot_and_region_value_df = time_slot_and_region_value_df.merge( pd.DataFrame(weekday_array, columns = ['weekday']), how = 'inner', left_index = True, right_index = True)
            new_time_slot_and_region_value_df = new_time_slot_and_region_value_df.reindex( columns = ['region', 'time_slot', 'weekday'] )

            # print('new_time_slot_and_region_value_df:')
            # print(new_time_slot_and_region_value_df)

            test_final_df = new_time_slot_and_region_value_df.merge(new_test_temp_df, how = 'inner', left_index = True, right_index = True)
            # print('test_final_df:')
            # print(test_final_df)

        else:

            temp_test_array = test_array[:, x, y, :]
            # print(temp_test_array.shape)
            
            new_test_temp_df = pd.DataFrame(temp_test_array, columns = ['arriving', 'departing'])

            # time_slot_and_region_value_array = np.mgrid[0 : time_slot_num : 1, major_station_num].reshape(2, -1).T
            # print(time_slot_and_region_value_array)

            time_slot_array = np.arange(test_time_slot_num)
            region_array = np.repeat(region, test_time_slot_num)

            # print(time_slot_array.shape)
            # print(region_array.shape)

            weekday_counter = 1

            weekday_counter -= 1
            weekday_array = np.zeros(test_time_slot_num, dtype = 'int32')
            for day_index in range( len(weekday_array) ):
                if day_index > 0 and day_index % 48 == 0:
                    weekday_counter = ( weekday_counter + 1) % 7
                weekday_array[day_index] = weekday_counter
            weekday_array += 1
            # weekday_array = np.repeat(weekday_array, major_station_num, axis = 0)

            time_slot_and_region_value_df = pd.DataFrame({'time_slot': time_slot_array, 'region': region_array})

            new_time_slot_and_region_value_df = time_slot_and_region_value_df.merge( pd.DataFrame(weekday_array, columns = ['weekday']), how = 'inner', left_index = True, right_index = True)
            new_time_slot_and_region_value_df = new_time_slot_and_region_value_df.reindex( columns = ['region', 'time_slot', 'weekday'] )

            # print('new_time_slot_and_region_value_df:')
            # print(new_time_slot_and_region_value_df)

            final_temp_df = new_time_slot_and_region_value_df.merge(new_test_temp_df, how = 'inner', left_index = True, right_index = True)
            # print('final_temp_df:')
            # print(final_temp_df)
            test_final_df = pd.concat([test_final_df, final_temp_df])
            # print('test_final_df:')
            # print(test_final_df)

test_final_df.reset_index(drop = True, inplace = True)
print('test_final_df:')
print(test_final_df)

train_final_df.to_csv('csv_files/ARIMA_training.csv')
test_final_df.to_csv('csv_files/ARIMA_testing.csv')
