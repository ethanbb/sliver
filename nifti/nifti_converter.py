import nibabel as nib
import numpy as np
import pdb

# library: http://nipy.org/nibabel/gettingstarted.html

data_folder = '/ihome/azhu/cs189/data/liverScans/Training Batch 1/'
num_examples = 28


def nifti_to_nparray(filename):
    img = nib.load(filename)
    data = img.get_data()
    return data


def convert_and_save_batch(folder, data_type):
    for i in range(0, num_examples):
        print 'converting ' + data_type + ' ' + str(i)
        file_name = folder + data_type + '-' + str(i)
        data = nifti_to_nparray(file_name + '.nii')

        # Concatenate an end-token at the end of the volume/segmentation
        x = data.shape[1]
        y = data.shape[0]
        end_token = np.ones((y, x, 1)) * -1
        data = np.concatenate((data, end_token), 2)

        # Save the np array
        npy_name = folder + 'npy_data/' + data_type + '-' + str(i)
        np.save(npy_name, data)
        print npy_name + ' saved'


if __name__ == '__main__':
    vol = convert_and_save_batch(data_folder, 'volume')
    seg = convert_and_save_batch(data_folder, 'segmentation')
