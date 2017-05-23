import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, labels, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

       Modified From:
           https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    """
    assert len(image.shape) == 2

    label0 = labels[..., 0]
    label1 = labels[..., 1]
    label2 = labels[..., 2]

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    image_mapped = map_coordinates(image, indices, order=1).reshape(shape)
    label0_mapped = map_coordinates(label0, indices, order=1).reshape(shape)
    label1_mapped = map_coordinates(label1, indices, order=1).reshape(shape)
    label2_mapped = map_coordinates(label2, indices, order=1).reshape(shape)

    labels[..., 0] = label0_mapped
    labels[..., 1] = label1_mapped
    labels[..., 2] = label2_mapped

    print('Performing elastic deformation')
    return image_mapped, labels
