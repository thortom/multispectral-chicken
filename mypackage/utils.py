import time
import numpy as np
from scipy.ndimage import measurements
import matplotlib.pyplot as plt

import mypackage

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        
def count_false_positive(Y_hat, Y_true, contaminant_numb=2, min_numb_pixels=0):
    def one_img(y_hat, y_true):
        array = (y_hat == contaminant_numb)*1

        # this defines the connection filter
        structure = np.ones((3, 3), dtype=np.int) # in this case we allow any kind of connection

        labeled, ncomponents = measurements.label(array, structure)
        indices = np.indices(array.shape).T[:,:,[1, 0]]

        fp_count = 0
        for i in range(1, ncomponents + 1):
            idx = indices[labeled == i]
            if (min_numb_pixels == 0) or (len(idx) > min_numb_pixels):
                fp_count += (contaminant_numb not in y_true[idx[:, 0], idx[:, 1]])*1

        return fp_count
    
    # TODO: This extra loop should not be needed
    if len(Y_true.shape) == 3:
        fp_count = one_img(np.squeeze(Y_hat), Y_true)
    else:
        fp_count = 0
        for i in range(len(Y_true)):
            fp_count += one_img(np.squeeze(Y_hat[i]), Y_true[i])
    return fp_count

def report_count_false_positive(Y_hat, Y_true):
    fp_count = count_false_positive(Y_hat, Y_true)
    print(f"Fasle positive blobs {fp_count}.")
    print(f"Fasle positive blobs per image {fp_count/len(Y_true):.4f}")
    
def plot_labeled_images(Y_hat, Y_true, plot_all=False):
    contaminant_numb = 2
    for i in range(len(Y_true)):
        if plot_all or (contaminant_numb in Y_true[i]):
            plt.figure(figsize=(9, 6))
            plt.subplot(121)
            img = plt.imshow(np.squeeze(Y_hat[i]))
            mypackage.Dataset._Dataset__add_legend_to_image(Y_hat[i], img)
            plt.title("Predicted labels")
            plt.subplot(122)
            img = plt.imshow(np.squeeze(Y_true[i]))
            mypackage.Dataset._Dataset__add_legend_to_image(Y_true[i], img)
            plt.title("True labels")
            plt.show()
            