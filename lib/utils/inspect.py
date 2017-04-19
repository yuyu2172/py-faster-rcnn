import numpy as np
import caffe


class Inspect(caffe.Layer):
    """
    Normalize all features
    """
    def setup(self, bottom, top):
        self.count = 0
        # Load layer param
        # layer_params = yaml.load(self.param_str)
        pass

    def reshape(self, bottom, top):
        bottom_shape = [x for x in bottom[0].data.shape]
        top[0].reshape(*bottom_shape)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        # Activation from the bottom layers
        bottom_data = bottom[0].data

        np.save('a.npy', bottom_data)

        print bottom_data.shape

    def backward(self, top, propagate_down, bottom):
        pass
