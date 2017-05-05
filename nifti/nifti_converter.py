import numpy as np
import nibabel as nib
import pdb

example_filename = './volume-1.nii'

img = nib.load(example_filename)
print 'here is the image'
print img.shape
print img.get_data_dtype()

data = img.get_data()
print 'here is the data'
print data1.shape
