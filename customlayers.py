import keras, tensorflow


class Grayscale(keras.engine.topology.Layer):

    def __init__(self, **kwargs):
        super(Grayscale, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Grayscale, self).build(input_shape)

    def call(self, x):
        return tensorflow.image.rgb_to_grayscale(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)
		

class Normalizer(keras.engine.topology.Layer):

    def __init__(self, **kwargs):
        super(Normalizer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Normalizer, self).build(input_shape)

    def call(self, x):
        return x / 127.5 - 1.

    def compute_output_shape(self, input_shape):
        return input_shape

		
def get():
    return {'Grayscale': Grayscale, 'Normalizer': Normalizer};

