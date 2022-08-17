import numpy as np
import pandas as pd

station_info_df = pd.read_csv("csv_files/station_info.csv")
print(station_info_df)

station_name_list = station_info_df["station_name"].tolist()
station_id_list = station_info_df["station_id"].tolist()
station_lat_list = station_info_df["station_lat"].tolist()
station_lng_list = station_info_df["station_lng"].tolist()

start_month = 7
end_month = 8
time_slot_num = (31 + 31) * 48
station_num = 1072

# 0 is inflow, 1 is outflow
arriving_and_departing_array = np.zeros( (time_slot_num, station_num, 2), dtype = 'int32')

for month_number in range(start_month, end_month + 1):
    print("Dealing with: csv_files/2020%02d-citibike-tripdata.csv" % month_number)
    month_df = pd.read_csv("csv_files/2020%02d-citibike-tripdata.csv" % month_number, low_memory = False)
    month_df = month_df.dropna(axis = 0, subset = ["start station name", "end station name", "start station id", "end station id"])
    # month_df["start station id"] = month_df["start station id"].astype(str)
    # month_df["end station id"] = month_df["end station id"].astype(str)

    for record_index, record in month_df.iterrows():

        # show the record_index
        print("reconrd_index:", record_index)

        # calculate start time slot index
        start_time = pd.to_datetime(record["starttime"], format = "%Y-%m-%d %H:%M:%S")
        start_time_day_index = 0

        if start_time.month == 7:
            start_time_day_index += 0
        elif start_time.month == 8:
            start_time_day_index += 31
        
        start_time_day_index += (start_time.day - 1)
        start_time_slot_index = (start_time_day_index * 48 + start_time.hour * 2 + (1 if start_time.minute >= 30 else 0))

        # calculate end time slot index
        end_time = pd.to_datetime(record["stoptime"], format = "%Y-%m-%d %H:%M:%S")
        end_time_day_index = 0

        if end_time.month == 7:
            end_time_day_index += 0
        elif end_time.month == 8:
            end_time_day_index += 31

        end_time_day_index += (end_time.day - 1)
        end_time_slot_index = (end_time_day_index * 48 + end_time.hour * 2 + (1 if end_time.minute >= 30 else 0))

        # find start station index
        start_station_index = station_id_list.index( record['start station id'] )

        # find end station index
        end_station_index = station_id_list.index( record['end station id'] )

        # outflow / departing
        arriving_and_departing_array[start_time_slot_index, start_station_index, 1] += 1

        # inflow / arriving
        arriving_and_departing_array[end_time_slot_index, end_station_index, 0] += 1

print('arriving_and_departing_array:')
print(arriving_and_departing_array)
print('arriving_and_departing_array.shape:')
print(arriving_and_departing_array.shape)

np.save('npy_files/arriving_and_departing_array.npy', arriving_and_departing_array)

new_arriving_and_departing_array = arriving_and_departing_array.reshape(-1, 2)
print('new_arriving_and_departing_array:')
print(new_arriving_and_departing_array)
print('new_arriving_and_departing_array.shape:')
print(new_arriving_and_departing_array.shape)

new_arriving_departing_df = pd.DataFrame(new_arriving_and_departing_array, columns = ['arriving', 'departing'])

time_slot_and_station_index_value_array = np.mgrid[0 : time_slot_num : 1, 0 : station_num : 1].reshape(2, -1).T
print(time_slot_and_station_index_value_array)

weekday_counter = 3

weekday_counter -= 1
weekday_array = np.zeros(time_slot_num, dtype = 'int32')
for day_index in range( len(weekday_array) ):
    if day_index > 0 and day_index % 48 == 0:
        weekday_counter = ( weekday_counter + 1) % 7
    weekday_array[day_index] = weekday_counter
weekday_array += 1
weekday_array = np.repeat(weekday_array, station_num, axis = 0)

time_slot_and_station_index_value_df = pd.DataFrame(time_slot_and_station_index_value_array, columns = ['time_slot', 'station_index'])

new_time_slot_and_station_index_value_df = time_slot_and_station_index_value_df.merge( pd.DataFrame(weekday_array, columns = ['weekday']), how = 'inner', left_index = True, right_index = True)
new_time_slot_and_station_index_value_df = new_time_slot_and_station_index_value_df.reindex( columns = ['time_slot', 'weekday', 'station_index'] )

print('new_time_slot_and_station_index_value_df:')
print(new_time_slot_and_station_index_value_df)

final_df = new_time_slot_and_station_index_value_df.merge(new_arriving_departing_df, how = 'inner', left_index = True, right_index = True)
print('final_df:')
print(final_df)

final_df.to_csv('csv_files/arriving_and_departing.csv')
