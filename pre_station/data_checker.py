import numpy as np
import pandas as pd

major_flow_station_info_df = pd.read_csv( 'csv_files/major_flow_station_info.csv' )
print(major_flow_station_info_df)

major_flow_df = pd.read_csv( 'csv_files/major_flow.csv' )
print(major_flow_df)

scalar_inflow_and_outflow_array = np.load('npy_files/scalar_inflow_and_outflow_array.npy')
scalar_volume_array = np.load('npy_files/scalar_volume_array.npy')
grid_inflow_and_outflow_array = np.load('npy_files/grid_inflow_and_outflow_array.npy')
grid_volume_array = np.load('npy_files/grid_volume_array.npy')

training_scalar_inflow_and_outflow_array = np.load('npy_files/training_scalar_inflow_and_outflow_array.npy')
testing_scalar_inflow_and_outflow_array = np.load('npy_files/testing_scalar_inflow_and_outflow_array.npy')

training_scalar_volume_array = np.load('npy_files/training_scalar_volume_array.npy')
testing_scalar_volume_array = np.load('npy_files/testing_scalar_volume_array.npy')

training_grid_inflow_and_outflow_array = np.load('npy_files/training_grid_inflow_and_outflow_array.npy')
testing_grid_inflow_and_outflow_array = np.load('npy_files/testing_grid_inflow_and_outflow_array.npy')

training_grid_volume_array = np.load('npy_files/training_grid_volume_array.npy')
testing_grid_volume_array = np.load('npy_files/testing_grid_volume_array.npy')

print('scalar_inflow_and_outflow_array.shape:')
print(scalar_inflow_and_outflow_array.shape)
print('scalar_volume_array.shape:')
print(scalar_volume_array.shape)
print('grid_inflow_and_outflow_array.shape:')
print(grid_inflow_and_outflow_array.shape)
print('grid_volume_array.shape:')
print(grid_volume_array.shape)

# print('scalar_volume_array.max():', scalar_volume_array.max())
# print('scalar_volume_array.min():', scalar_volume_array.min())

# print('grid_volume_array.max():', grid_volume_array.max())
# print('grid_volume_array.min():', grid_volume_array.min())

# print('scalar_inflow_and_outflow_array.max():', scalar_inflow_and_outflow_array.max())
# print('scalar_inflow_and_outflow_array.min():', scalar_inflow_and_outflow_array.min())

# print('grid_inflow_and_outflow_array.max():', grid_inflow_and_outflow_array.max())
# print('grid_inflow_and_outflow_array.min():', grid_inflow_and_outflow_array.min())

print()
print('training_scalar_inflow_and_outflow_array.max():', training_scalar_inflow_and_outflow_array.max())
print('training_scalar_inflow_and_outflow_array.min():', training_scalar_inflow_and_outflow_array.min())
print()
print('testing_scalar_inflow_and_outflow_array.max():', testing_scalar_inflow_and_outflow_array.max())
print('testing_scalar_inflow_and_outflow_array.min():', testing_scalar_inflow_and_outflow_array.min())
print()
print('training_scalar_volume_array.max():', training_scalar_volume_array.max())
print('training_scalar_volume_array.min():', training_scalar_volume_array.min())
print()
print('testing_scalar_volume_array.max():', testing_scalar_volume_array.max())
print('testing_scalar_volume_array.min():', testing_scalar_volume_array.min())
print()
print('training_grid_inflow_and_outflow_array.max():', training_grid_inflow_and_outflow_array.max())
print('training_grid_inflow_and_outflow_array.min():', training_grid_inflow_and_outflow_array.min())
print()
print('testing_grid_inflow_and_outflow_array.max():', testing_grid_inflow_and_outflow_array.max())
print('testing_grid_inflow_and_outflow_array.min():', testing_grid_inflow_and_outflow_array.min())
print()
print('training_grid_volume_array.max():', training_grid_volume_array.max())
print('training_grid_volume_array.min():', training_grid_volume_array.min())
print()
print('testing_grid_volume_array.max():', testing_grid_volume_array.max())
print('testing_grid_volume_array.min():', testing_grid_volume_array.min())
