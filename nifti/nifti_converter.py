import nibabel as nib
import numpy as np
import os
import pdb


# library: http://nipy.org/nibabel/gettingstarted.html

example_filename = './volume-1.nii'


def nifti_to_nparray(filename):
    img = nib.load(filename)
    data = img.get_data()
    return data


def convert_batch_volumes(folder):
    nifti_files = []
