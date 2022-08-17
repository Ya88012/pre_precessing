import numpy as np

vdata_array = np.load('npy_files/vdata.npy')
fdata_array = np.load('npy_files/fdata.npy')

print('vdata_array.shape:')
print(vdata_array.shape)
print('fdata_array.shape:')
print(fdata_array.shape)

need_range = 60 * 48

vdata_array = vdata_array[: need_range]
fdata_array = fdata_array[:, : need_range]

training_dataset_range = 40 * 48

# vdata_array
training_vdata_array = vdata_array[: training_dataset_range]
testing_vdata_array = vdata_array[training_dataset_range :]
# fdata_array
training_fdata_array = fdata_array[:, : training_dataset_range]
testing_fdata_array = fdata_array[:, training_dataset_range :]

print('training_vdata_array.shape:', training_vdata_array.shape)
print('testing_vdata_array.shape:', testing_vdata_array.shape)
print('training_fdata_array.shape:', training_fdata_array.shape)
print('testing_fdata_array.shape:', testing_fdata_array.shape)

np.save('npy_files/training_vdata_array.npy', training_vdata_array)
np.save('npy_files/testing_vdata_array.npy', testing_vdata_array)
np.save('npy_files/training_fdata_array.npy', training_fdata_array)
np.save('npy_files/testing_fdata_array.npy', testing_fdata_array)
