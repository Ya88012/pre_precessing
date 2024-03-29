import numpy as np
import pandas as pd

train_array = np.load('npy_files/training_scalar_volume_array.npy')
test_array = np.load('npy_files/testing_scalar_volume_array.npy')

train_time_slot_num = (40) * 48
test_time_slot_num = (20) * 48
major_station_num = 100

for major_flow_station_index in range(major_station_num):

    if major_flow_station_index == 0:

        temp_train_array = train_array[:, major_flow_station_index, :]
        # print(temp_train_array)
        
        new_train_temp_df = pd.DataFrame(temp_train_array, columns = ['start', 'end'])
        # print(new_train_temp_df)

        # time_slot_and_station_index_value_array = np.mgrid[0 : time_slot_num : 1, major_station_num].reshape(2, -1).T
        # print(time_slot_and_station_index_value_array)

        time_slot_array = np.arange(train_time_slot_num)
        station_index_array = np.repeat(major_flow_station_index, train_time_slot_num)

        # print(time_slot_array.shape)
        # print(station_index_array.shape)

        weekday_counter = 3

        weekday_counter -= 1
        weekday_array = np.zeros(train_time_slot_num, dtype = 'int32')
        for day_index in range( len(weekday_array) ):
            if day_index > 0 and day_index % 48 == 0:
                weekday_counter = ( weekday_counter + 1) % 7
            weekday_array[day_index] = weekday_counter
        weekday_array += 1
        # weekday_array = np.repeat(weekday_array, major_station_num, axis = 0)

        time_slot_and_station_index_value_df = pd.DataFrame({'time_slot': time_slot_array, 'station_index': station_index_array})

        new_time_slot_and_station_index_value_df = time_slot_and_station_index_value_df.merge( pd.DataFrame(weekday_array, columns = ['weekday']), how = 'inner', left_index = True, right_index = True)
        new_time_slot_and_station_index_value_df = new_time_slot_and_station_index_value_df.reindex( columns = ['station_index', 'time_slot', 'weekday'] )

        # print('new_time_slot_and_station_index_value_df:')
        # print(new_time_slot_and_station_index_value_df)

        train_final_df = new_time_slot_and_station_index_value_df.merge(new_train_temp_df, how = 'inner', left_index = True, right_index = True)
        # print('train_final_df:')
        # print(train_final_df)

    else:

        temp_train_array = train_array[:, major_flow_station_index, :]
        # print(temp_train_array.shape)
        
        new_train_temp_df = pd.DataFrame(temp_train_array, columns = ['start', 'end'])

        # time_slot_and_station_index_value_array = np.mgrid[0 : time_slot_num : 1, major_station_num].reshape(2, -1).T
        # print(time_slot_and_station_index_value_array)

        time_slot_array = np.arange(train_time_slot_num)
        station_index_array = np.repeat(major_flow_station_index, train_time_slot_num)

        # print(time_slot_array.shape)
        # print(station_index_array.shape)

        weekday_counter = 3

        weekday_counter -= 1
        weekday_array = np.zeros(train_time_slot_num, dtype = 'int32')
        for day_index in range( len(weekday_array) ):
            if day_index > 0 and day_index % 48 == 0:
                weekday_counter = ( weekday_counter + 1) % 7
            weekday_array[day_index] = weekday_counter
        weekday_array += 1
        # weekday_array = np.repeat(weekday_array, major_station_num, axis = 0)

        time_slot_and_station_index_value_df = pd.DataFrame({'time_slot': time_slot_array, 'station_index': station_index_array})

        new_time_slot_and_station_index_value_df = time_slot_and_station_index_value_df.merge( pd.DataFrame(weekday_array, columns = ['weekday']), how = 'inner', left_index = True, right_index = True)
        new_time_slot_and_station_index_value_df = new_time_slot_and_station_index_value_df.reindex( columns = ['station_index', 'time_slot', 'weekday'] )

        # print('new_time_slot_and_station_index_value_df:')
        # print(new_time_slot_and_station_index_value_df)

        final_temp_df = new_time_slot_and_station_index_value_df.merge(new_train_temp_df, how = 'inner', left_index = True, right_index = True)
        # print('final_temp_df:')
        # print(final_temp_df)
        train_final_df = pd.concat([train_final_df, final_temp_df])
        # print('train_final_df:')
        # print(train_final_df)

train_final_df.reset_index(drop = True, inplace = True)
print('train_final_df:')
print(train_final_df)

for major_flow_station_index in range(major_station_num):

    if major_flow_station_index == 0:

        temp_test_array = test_array[:, major_flow_station_index, :]
        # print(temp_test_array.shape)
        
        new_test_temp_df = pd.DataFrame(temp_test_array, columns = ['start', 'end'])

        # time_slot_and_station_index_value_array = np.mgrid[0 : time_slot_num : 1, major_station_num].reshape(2, -1).T
        # print(time_slot_and_station_index_value_array)

        time_slot_array = np.arange(test_time_slot_num)
        station_index_array = np.repeat(major_flow_station_index, test_time_slot_num)

        # print(time_slot_array.shape)
        # print(station_index_array.shape)

        weekday_counter = 1

        weekday_counter -= 1
        weekday_array = np.zeros(test_time_slot_num, dtype = 'int32')
        for day_index in range( len(weekday_array) ):
            if day_index > 0 and day_index % 48 == 0:
                weekday_counter = ( weekday_counter + 1) % 7
            weekday_array[day_index] = weekday_counter
        weekday_array += 1
        # weekday_array = np.repeat(weekday_array, major_station_num, axis = 0)

        time_slot_and_station_index_value_df = pd.DataFrame({'time_slot': time_slot_array, 'station_index': station_index_array})

        new_time_slot_and_station_index_value_df = time_slot_and_station_index_value_df.merge( pd.DataFrame(weekday_array, columns = ['weekday']), how = 'inner', left_index = True, right_index = True)
        new_time_slot_and_station_index_value_df = new_time_slot_and_station_index_value_df.reindex( columns = ['station_index', 'time_slot', 'weekday'] )

        # print('new_time_slot_and_station_index_value_df:')
        # print(new_time_slot_and_station_index_value_df)

        test_final_df = new_time_slot_and_station_index_value_df.merge(new_test_temp_df, how = 'inner', left_index = True, right_index = True)
        # print('test_final_df:')
        # print(test_final_df)

    else:

        temp_test_array = test_array[:, major_flow_station_index, :]
        # print(temp_test_array.shape)
        
        new_test_temp_df = pd.DataFrame(temp_test_array, columns = ['start', 'end'])

        # time_slot_and_station_index_value_array = np.mgrid[0 : time_slot_num : 1, major_station_num].reshape(2, -1).T
        # print(time_slot_and_station_index_value_array)

        time_slot_array = np.arange(test_time_slot_num)
        station_index_array = np.repeat(major_flow_station_index, test_time_slot_num)

        # print(time_slot_array.shape)
        # print(station_index_array.shape)

        weekday_counter = 1

        weekday_counter -= 1
        weekday_array = np.zeros(test_time_slot_num, dtype = 'int32')
        for day_index in range( len(weekday_array) ):
            if day_index > 0 and day_index % 48 == 0:
                weekday_counter = ( weekday_counter + 1) % 7
            weekday_array[day_index] = weekday_counter
        weekday_array += 1
        # weekday_array = np.repeat(weekday_array, major_station_num, axis = 0)

        time_slot_and_station_index_value_df = pd.DataFrame({'time_slot': time_slot_array, 'station_index': station_index_array})

        new_time_slot_and_station_index_value_df = time_slot_and_station_index_value_df.merge( pd.DataFrame(weekday_array, columns = ['weekday']), how = 'inner', left_index = True, right_index = True)
        new_time_slot_and_station_index_value_df = new_time_slot_and_station_index_value_df.reindex( columns = ['station_index', 'time_slot', 'weekday'] )

        # print('new_time_slot_and_station_index_value_df:')
        # print(new_time_slot_and_station_index_value_df)

        final_temp_df = new_time_slot_and_station_index_value_df.merge(new_test_temp_df, how = 'inner', left_index = True, right_index = True)
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
