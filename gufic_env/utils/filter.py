import numpy as np
from scipy import signal

## Originally written in learn-seq git repo:
## https://github.com/deanpham98/learn-seq

class ButterLowPass:
    """
    butterworth lowpass digital filter design.

    Check C{scipy.signal.butter} for further details.
    """
    def __init__(self, cutoff, fs, order=5):
        """
        C{ButterLowPass} constructor

        @type  cutoff: float
        @param cutoff: Cut-off frequency in Hz
        @type  fs: float
        @param fs: The sampling frequency (Hz) of the signal to be filtered
        @type  order: int
        @param order: The order of the filter.
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = signal.butter(order,
                                       normal_cutoff,
                                       btype='low',
                                       analog=False)

    def __call__(self, x):
        """
        Filters the input array across its C{axis=0} (each column is
        considered as an independent signal). Uses initial conditions (C{zi})
        for the filter delays.

        @type x: array
        @param x: An N-dimensional input array.
        @rtype: array
        @return: The output of the digital filter.
        """
        if not hasattr(self, 'zi'):
            cols = x.shape[1]
            # zi = signal.lfiltic(self.b, self.a, []).tolist() * cols
            self.zi = np.array(
                signal.lfiltic(self.b, self.a, []).tolist() * cols)
            self.zi.shape = (-1, cols)
        (filtered, self.zi) = signal.lfilter(self.b,
                                             self.a,
                                             x,
                                             zi=self.zi,
                                             axis=0)
        return filtered

    def reset_state(self):
        if hasattr(self, 'zi'):
            delattr(self, 'zi')