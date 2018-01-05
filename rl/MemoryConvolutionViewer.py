from rl.MemoryLogger import MemoryLogger
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


class MemoryConvolutionViewer(MemoryLogger):
    def __init__(self, memory, graph, conv_model, width=224, height=256):
        super().__init__(memory)
        plt.ion()
        self._model = conv_model
        self._graph = graph
        self._image = None
        self._image_buffer = np.zeros(shape=(width, height))

    def redraw(self, frames):
        with self._graph.as_default():
            conv_frames = self._model.predict(frames)[0]

            num_small_images = conv_frames.shape[0]
            num_images = int(np.math.sqrt(num_small_images))
            if num_images * num_images < num_small_images:
                num_images += 1

            x_offset = 0
            y_offset = 0
            im_width = self._image_buffer.shape[0] // num_images
            im_height = self._image_buffer.shape[1] // num_images

            for row in range(num_images):
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
                y_offset = 0

            if self._image is None:
                self._image = plt.imshow(self._image_buffer)
            else:
                self._image.set_data(self._image_buffer)
            plt.draw()
            plt.pause(0.0001)

    def add(self, state, reward, scores, next_state, is_final_state):
        super().add(state, reward, scores, next_state, is_final_state)

        self._debug_logger_thread.run_on_thread(self.redraw, state)
