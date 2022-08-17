from math import floor
import numpy as np
import pandas as pd

from GPS_utils import gps_to_xy

# station_info
station_info_df = pd.read_csv("csv_files/station_info.csv")
print(station_info_df)

station_name_list = station_info_df["station_name"].tolist()
station_id_list = station_info_df["station_id"].tolist()
station_lat_list = station_info_df["station_lat"].tolist()
station_lng_list = station_info_df["station_lng"].tolist()

station_num = len(station_info_df)
time_slot_num = ( 31 + 31 ) * 48

start_month = 7
end_month = 8

w = 10
h = 20

# 0 is x, 1 is y.
# create an array to keep all stations grid coordinates
coor_array = np.full( (station_num, 2), -1, dtype = 'float64')

# crate [v]olume data array and [f]low data array
vdata = np.zeros( (time_slot_num, w, h, 2), dtype = 'int32')
fdata = np.zeros( (2, time_slot_num, w, h, w, h), dtype = 'int32' )

for i in range(station_num):
    c = gps_to_xy(station_lng_list[i], station_lat_list[i])
    coor_array[i, 0] = c[0]
    coor_array[i, 1] = c[1]

print('coor_array[:, 0].max():', coor_array[:, 0].max())
print('coor_array[:, 0].max():', coor_array[:, 0].min())
print('coor_array[:, 0].max():', coor_array[:, 1].max())
print('coor_array[:, 0].max():', coor_array[:, 1].min())

for month_number in range(start_month, end_month + 1):

    print("Dealing with: csv_files/2020%02d-citibike-tripdata.csv" % month_number)
    month_df = pd.read_csv("csv_files/2020%02d-citibike-tripdata.csv" % month_number, low_memory = False)
    month_df = month_df.dropna(axis = 0, subset = ["start station name", "end station name", "start station id", "end station id"])
    # month_df["start_station_id"] = month_df["start_station_id"].astype(str)
    # month_df["end_station_id"] = month_df["end_station_id"].astype(str)

    for record_index, record in month_df.iterrows():

        print("record_index:", record_index)

        start_station_index = station_id_list.index( record["start station id"] )
        end_station_index = station_id_list.index( record["end station id"] )
        start_time = pd.to_datetime( record["starttime"], format = "%Y-%m-%d %H:%M:%S" )
        end_time = pd.to_datetime( record["stoptime"], format = "%Y-%m-%d %H:%M:%S" )

        start_time_day_index = 0
        if start_time.month == 7:
            start_time_day_index += 0
        elif start_time.month == 8:
            start_time_day_index += 31
        start_time_day_index += (start_time.day - 1)
        start_time_slot_index = (start_time_day_index * 48 + start_time.hour * 2 + (1 if start_time.minute >= 30 else 0))

        end_time_day_index = 0
        if end_time.month == 7:
            end_time_day_index += 0
        elif end_time.month == 8:
            end_time_day_index += 31
        end_time_day_index += (end_time.day - 1)
        end_time_slot_index = (end_time_day_index * 48 + end_time.hour * 2 + (1 if end_time.minute >= 30 else 0))

        # starts_inside, ends_inside: Booleans.
        # True if the trip starts within Manhattan, false otherwise
        starts_inside = (0 <= coor_array[start_station_index, 0] <= 1) and (0 <= coor_array[start_station_index, 1] <= 1)
        ends_inside   = (0 <= coor_array[end_station_index, 0] <= 1) and (0 <= coor_array[end_station_index, 1] <= 1)

        starts_and_ends_in_same_month = (start_time.month == end_time.month)

        # Variable names:
        #   s/e stands for start/end, g stands for grid, x/y are coordinates
        sgx = floor(coor_array[start_station_index, 0] * w) #start-x, mapped to grid coordinates
        sgy = floor(coor_array[start_station_index, 1] * h) #start-y, mapped to grid coordinates
        egx = floor(coor_array[end_station_index, 0] * w) #end-x, mapped to grid coordinates
        egy = floor(coor_array[end_station_index, 1] * h) #end-y, mapped to grid coordinates

        if starts_inside:
            # Update volume data for the start of the trip
            vdata[start_time_slot_index, sgx, sgy, 0] += 1
            
            if ends_inside:
                # Update flow data only if the trip starts and ends within Manhattan.
                if start_time_slot_index == end_time_slot_index:
                    fdata[0, end_time_slot_index, sgx, sgy, egx, egy] += 1
                else:
                    fdata[1, end_time_slot_index, sgx, sgy, egx, egy] += 1
                    
        if ends_inside:
            # Update volume data for the end of the trip.
            vdata[end_time_slot_index, egx, egy, 1] += 1

print('vdata:')
print(vdata)
print('vdata.shape:')
print(vdata.shape)
print('fdata:')
print(fdata)
print('fdata.shape:')
print(fdata.shape)

np.save('npy_files/vdata.npy', vdata)
np.save('npy_files/fdata.npy', fdata)
