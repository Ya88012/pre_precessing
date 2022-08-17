import numpy as np

scalar_inflow_and_outflow_array = np.load('npy_files/scalar_inflow_and_outflow_array.npy')
scalar_volume_array = np.load('npy_files/scalar_volume_array.npy')
grid_inflow_and_outflow_array = np.load('npy_files/grid_inflow_and_outflow_array.npy')
grid_volume_array = np.load('npy_files/grid_volume_array.npy')

print('scalar_inflow_and_outflow_array.shape:')
print(scalar_inflow_and_outflow_array.shape)
print('scalar_volume_array.shape:')
print(scalar_volume_array.shape)
print('grid_inflow_and_outflow_array.shape:')
print(grid_inflow_and_outflow_array.shape)
print('grid_volume_array.shape:')
print(grid_volume_array.shape)

need_range = 60 * 48

scalar_inflow_and_outflow_array = scalar_inflow_and_outflow_array[ : need_range]
scalar_volume_array = scalar_volume_array[ : need_range]
grid_inflow_and_outflow_array = grid_inflow_and_outflow_array[ : need_range]
grid_volume_array = grid_volume_array[ : need_range]

training_dataset_range = 40 * 48

# scalar_inflow_and_outflow_array
training_scalar_inflow_and_outflow_array = scalar_inflow_and_outflow_array[ : training_dataset_range]
testing_scalar_inflow_and_outflow_array = scalar_inflow_and_outflow_array[training_dataset_range : ]
# scalar_volume_array
training_scalar_volume_array = scalar_volume_array[ : training_dataset_range]
testing_scalar_volume_array = scalar_volume_array[training_dataset_range : ]
# grid_inflow_and_outflow_array
training_grid_inflow_and_outflow_array = grid_inflow_and_outflow_array[ : training_dataset_range]
testing_grid_inflow_and_outflow_array = grid_inflow_and_outflow_array[training_dataset_range : ]
# grid_volume_array
training_grid_volume_array = grid_volume_array[ : training_dataset_range]
testing_grid_volume_array = grid_volume_array[training_dataset_range : ]

print('training_scalar_inflow_and_outflow_array.shape:', training_scalar_inflow_and_outflow_array.shape)
print('testing_scalar_inflow_and_outflow_array.shape:', testing_scalar_inflow_and_outflow_array.shape)
print('training_scalar_volume_array.shape:', training_scalar_volume_array.shape)
print('testing_scalar_volume_array.shape:', testing_scalar_volume_array.shape)
print('training_grid_inflow_and_outflow_array.shape:', training_grid_inflow_and_outflow_array.shape)
print('testing_grid_inflow_and_outflow_array.shape:', testing_grid_inflow_and_outflow_array.shape)
print('training_grid_volume_array.shape:', training_grid_volume_array.shape)
print('testing_grid_volume_array.shape:', testing_grid_volume_array.shape)

np.save('npy_files/training_scalar_inflow_and_outflow_array.npy', training_scalar_inflow_and_outflow_array)
np.save('npy_files/testing_scalar_inflow_and_outflow_array.npy', testing_scalar_inflow_and_outflow_array)
np.save('npy_files/training_scalar_volume_array.npy', training_scalar_volume_array)
np.save('npy_files/testing_scalar_volume_array.npy', testing_scalar_volume_array)
np.save('npy_files/training_grid_inflow_and_outflow_array.npy', training_grid_inflow_and_outflow_array)
np.save('npy_files/testing_grid_inflow_and_outflow_array.npy', testing_grid_inflow_and_outflow_array)
np.save('npy_files/training_grid_volume_array.npy', training_grid_volume_array)
np.save('npy_files/testing_grid_volume_array.npy', testing_grid_volume_array)
