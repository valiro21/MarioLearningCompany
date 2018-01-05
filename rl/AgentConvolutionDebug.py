import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras import backend as K


class AgentConvolutionDebug(object):
    def __init__(self, agent, debug_logger_thread, layer=0, width=224, height=256):
        self.__class__ = type(agent.__class__.__name__,
                              (self.__class__, agent.__class__),
                              {})
        self.__dict__ = agent.__dict__

        plt.ion()
        self._agent = agent
        self._debug_logger_thread = debug_logger_thread
        self._layer = layer
        self._image = None
        self._image_buffer = np.zeros(shape=(width, height))

    def _compute_scores(self, observation):
        model = self._agent.model

        get_outputs = K.function([model.layers[0].input, K.learning_phase()],
                                 [model.layers[self._layer].output, model.layers[-1].output])
        conv_frames, score = get_outputs([observation, 0])

        self._debug_logger_thread.run_on_thread(self.redraw, conv_frames[0])

        return score[0]

    def redraw(self, conv_frames):
        num_small_images = conv_frames.shape[0]
        num_images = int(np.math.sqrt(num_small_images))
        if num_images * num_images < num_small_images:
            num_images += 1

        im_width = self._image_buffer.shape[0] // num_images
        im_height = self._image_buffer.shape[1] // num_images

        x_offset = 0
        for row in range(num_images):
            y_offset = 0
            for col in range(num_images):
                if num_images * row + col >= num_small_images:
                    break

                imageholder = self._image_buffer[
                                x_offset:x_offset+im_width,
                                y_offset:y_offset+im_height
                              ]

                max_value = np.max(conv_frames[num_images * row + col])
                imageholder[:] = resize(conv_frames[num_images * row + col] / max_value, (im_width, im_height))

                y_offset += im_height
            x_offset += im_width

        if self._image is None:
            self._image = plt.imshow(self._image_buffer)
        else:
            self._image.set_data(self._image_buffer)
        plt.draw()
        plt.pause(0.0001)
