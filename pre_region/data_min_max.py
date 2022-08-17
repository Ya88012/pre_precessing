import numpy as np

training_vdata_array = np.load('npy_files/training_vdata_array.npy')
testing_vdata_array = np.load('npy_files/testing_vdata_array.npy')
training_fdata_array = np.load('npy_files/training_fdata_array.npy')
testing_fdata_array = np.load('npy_files/testing_fdata_array.npy')

print('training_vdata_array.max():', training_vdata_array.max())
print('training_vdata_array.min():', training_vdata_array.min())

print('testing_vdata_array.max():', testing_vdata_array.max())
print('testing_vdata_array.min():', testing_vdata_array.min())

print('training_fdata_array.max():', training_fdata_array.max())
print('training_fdata_array.min():', training_fdata_array.min())

print('testing_fdata_array.max():', testing_fdata_array.max())
print('testing_fdata_array.min():', testing_fdata_array.min())
