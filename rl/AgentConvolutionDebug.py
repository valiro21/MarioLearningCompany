import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras import backend as K

from rl.CustomEnv import normalize


class AgentConvolutionDebug(object):
    def __init__(self, agent, debug_logger_thread,
                 layers=[0], show_network_input=True,
                 width=224, height=256):
        self.__class__ = type(agent.__class__.__name__,
                              (self.__class__, agent.__class__),
                              {})
        self.__dict__ = agent.__dict__

        self._agent = agent
        self._debug_logger_thread = debug_logger_thread
        self._layers = layers
        self._num_images = len(layers) + (1 if show_network_input else 0)
        self._images = None
        self._images_buffer = np.zeros(shape=(self._num_images, width, height))
        self._show_network_input = show_network_input

    def _compute_scores(self, observation):
        model = self._agent.model

        outputs = [model.layers[layer].output for layer in self._layers]
        outputs = outputs + [model.layers[-1].output]
        get_outputs = K.function([model.layers[0].input, K.learning_phase()],
                                 outputs)
        model_outputs = get_outputs([observation, 0])

        to_draw = model_outputs[:-1]
        if self._show_network_input:
            to_draw = [observation] + to_draw
        self._debug_logger_thread.run_on_thread(self.redraw, to_draw)

        return model_outputs[-1][0]

    def redraw(self, convs_frames):
        for idx, conv_frames in enumerate(convs_frames):
            conv_frames = conv_frames[0]
            num_small_images = conv_frames.shape[0]
            num_images = int(np.math.sqrt(num_small_images))
            if num_images * num_images < num_small_images:
                num_images += 1

            im_width = self._images_buffer.shape[1] // num_images
            im_height = self._images_buffer.shape[2] // num_images

            x_offset = 0
            for row in range(num_images):
                y_offset = 0
                for col in range(num_images):
                    if num_images * row + col >= num_small_images:
                        break
                    imageholder = self._images_buffer[
                                    idx,
                                    x_offset:x_offset+im_width,
                                    y_offset:y_offset+im_height
                                  ]

                    max_value = np.max(conv_frames[num_images * row + col])
                    imageholder[:] = resize(normalize(conv_frames[num_images * row + col]), (im_width, im_height))

                    y_offset += im_height
                x_offset += im_width

            if self._images is None:
                plt.ion()
                self._images = [None] * self._num_images

            if self._images[idx] is None:
                plt.figure()
                self._images[idx] = plt.imshow(self._images_buffer[idx])
            else:
                self._images[idx].set_data(self._images_buffer[idx])
            plt.draw()
            plt.pause(0.0001)
