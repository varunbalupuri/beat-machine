import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.signal as sps
import scipy.ndimage
import audioop
import logging
import os
import shutil
import wave

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataProcessor(object):
    """
    Contains the methods required for turning
    raw audio data into spectrogram arrays
    for use in neural network. Also contains
    data visualisation and introspection
    methods
    """

    def __init__(self, filepath, lowcut=500, highcut=15000,
                 fft_size=2048, spec_thresh=4, n_freq_components=64,
                 shorten_factor=10, start_freq=300,
                 end_freq=8000, samplerate=16000):
        """
        Args:
            # Required #
            filepath (str) : the filepath of the audio file (wav)
            # Optional #
            lowcut (int) : Low cut - butter bandpass filter
            highcut (int) : High cut - butter bandpass filter
            fft_size (int) : window size for the FFT
            spec_thresh (int) : threshold for spectrograms
            n_mel_freq_components (int) : number of mel frequency channels
            shorten_factor (int) : compression factor on the x-axis (time)
            start_freq (int) : Frequency to start sampling our melS from
            end_freq (int) : Frequency to stop sampling our melS from
        """
        self.filepath = filepath
        self.lowcut = lowcut
        self.highcut = highcut
        self.fft_size = fft_size
        self.step_size = int(fft_size/16)
        self.spec_thresh = spec_thresh
        self.n_freq_components = n_freq_components
        self.shorten_factor = shorten_factor
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.samplerate = samplerate

    ###############################
    #  loading and augmentation   #
    ###############################

    def _downsample(self, out_rate):
        """ Downsamples data at self.filepath to have rate, `outrate`
        and writes data to to file with new downsampled rate. Neccesary
        for dimensionality/complexity reduction in classification
        Args:
            out_rate (int) : the desired downsampled rate
        """
        temp_filepath = 'downsample_{}_{}'.format(out_rate, self.filepath)

        in_wav = wave.open(self.filepath)
        logger.info('original rate is : %s', in_wav.getframerate())

        out_wav = wave.open(temp_filepath, "w")
        out_wav.setframerate(out_rate)
        out_wav.setnchannels(in_wav.getnchannels())
        out_wav.setsampwidth(in_wav.getsampwidth())
        out_wav.setnframes(1)

        if in_wav.getsampwidth() == 1:
            nptype = np.uint8
        elif in_wav.getsampwidth() == 2:
            nptype = np.uint16

        in_rate = in_wav.getframerate()
        in_nframes = in_wav.getnframes()

        audio = in_wav.readframes(in_nframes)
        nroutsamples = round(len(audio) * out_rate/in_rate)

        audio_out = sps.resample(np.fromstring(audio, nptype), nroutsamples)
        audio_out = audio_out.astype(nptype)

        out_wav.writeframes(audio_out.copy(order='C'))

        logger.info('written {} with rate {}'.format(temp_filepath, out_rate))
        out_wav.close()

        shutil.copy(temp_filepath, self.filepath)

        logger.info('removing tempfile: {}'.format(temp_filepath))
        os.remove(temp_filepath)

        return out_rate

    def load_data(self, rate, n_secs):
        """Loads a n_sec sample of the file
        into instance attribtes, downsamples and then
        loads into self.rate and self.data
        Args:
            rate (int) : The desired rate with which to downsample the data to 
            n_secs (float) : The number of seconds to sample from any given file.
        Returns:
            None
        """
        new_rate = self._downsample(rate)
        logger.info('downsampling {} to {}'.format(self.filepath,new_rate))
        self.rate, data = wavfile.read(self.filepath)
        logger.info('rate read as: {}'.format(self.rate))

        # should not be too close to start/end of song
        # req_buffer seconds at start and end of track as no go zone
        req_buffer = 5
        buffer_zone = self.rate*(n_secs+req_buffer)
        random_start = random.randint(buffer_zone, len(data)-buffer_zone)

        data = data[random_start: self.rate*n_secs]

        self.data = data

    def _add_noise(self, mu, sigma):
        """Adds Gaussian noise to self.data
        with params mean mu, std_dev sigma.
        In the case of uniform noise, these are ignored
        Args:
            mu (float): mean of normal dist.
            sigma (float): std dev of normal dist.
        """
        if self.data is None:
            raise Exception('data is not loaded, use load_data method')

        self.data = list(np.array(self.data) +
                         np.random.normal(mu, sigma, len(data)))

    def _hz_to_mel(self, hz):
        """
        Convert a value in Hertz to Mels
        Args:
            hz (float) a value in Hz.
        Returns:
            (float) a value in Mels.
        """
        return 2595 * np.log10(1+hz/700.)

    def _mel_to_hz(self, mel):
        """
        Convert a value in Hertz to Mels
        Args:
            mel (float) a value in Mels.
        Returns:
            (float) a value in Hz.
        """
        return 700*(10**(mel/2595.0)-1)

    def butter_bandpass_filter(self, data, rate, order=5):
        """ Butterworth filter
        Args:
            data (np.array) : wav file data
            rate (float) : rate of the data
        Returns
            (np.array) butter bandpass filtered data
        """
        nyqist_freq = 0.5 * rate
        low = self.lowcut / nyqist_freq
        high = self.highcut / nyqist_freq
        b, a = butter(order, [low, high], btype='band')

        y = lfilter(b, a, data)
        return y

    def overlap(self, X, window_size, window_step):
        """
        Create an overlapped version of X
        Args:
            X (ndarray) : shape=(n_samples,) : Input signal to window and overlap
            window_size (int) : Size of windows to take
            window_step (int) : Step size between windows
        Returns
            X_strided (np.array shape=(n_windows, window_size)) 2D array of overlapped X
        """
        if window_size % 2 != 0:
            raise ValueError("Window size must be even!")
        # Make sure there are an even number of windows before stridetricks
        append = np.zeros((window_size - len(X) % window_size))
        X = np.hstack((X, append))

        ws = window_size
        ss = window_step
        a = X

        valid = len(a) - ws
        nw = int(valid / ss)
        out = np.ndarray((nw, ws), dtype=a.dtype)

        for i in range(nw):
            # "slide" the window along the samples
            start = int(i * ss)
            stop = int(start + ws)
            out[i] = a[start: stop]

        return out

    def stft(self, X, fftsize=128, step=65, mean_normalize=True, real=False,
             compute_onesided=True):
        """
        Calculates short time Fourier transform for 1D real valued input X
        Args:
            x (np.array) input data
        Returns:
            (np.array) stFT transformed data
        """
        if real:
            local_fft = np.fft.rfft
            cut = -1
        else:
            local_fft = np.fft.fft
            cut = None
        if compute_onesided:
            cut = fftsize // 2
        if mean_normalize:
            X -= X.mean()

        X = self.overlap(X, fftsize, step)

        size = fftsize
        win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
        X = X * win[None]
        X = local_fft(X)[:, :cut]
        return X

    def pretty_spectrogram(self, data):
        """
        Creates a spectrogram
        Args:
            data (np.array)
            log (bool) : Whether to log normalise the spectrgram
            thresh (float) : threshold minimum power for log spectrogram
        Returns:
            (np.array) Spectrogram representation of audio
        """
        fft_size = self.fft_size
        step_size = self.step_size
        log = True
        thresh = self.spec_thresh

        specgram = np.abs(self.stft(data, fftsize=fft_size,
                          step=step_size, real=False, compute_onesided=True))

        if log:
            specgram /= specgram.max()  # volume normalize to max 1
            specgram = np.log10(specgram)
            # set anything less than the threshold as the threshold
            specgram[specgram < -thresh] = -thresh
        else:
            # set anything less than the threshold as the threshold
            specgram[specgram < thresh] = thresh

        return specgram

    def make_mel(self, spectrogram, mel_filter, shorten_factor=1):
        mel_spec = np.transpose(mel_filter).dot(np.transpose(spectrogram))
        mel_spec = scipy.ndimage.zoom(mel_spec.astype('float32'),
                                      [1, 1./shorten_factor]).astype('float64')
        mel_spec = mel_spec[:, 1:-1]
        return mel_spec

    def get_filterbanks(self, nfilt=20, nfft=512,
                        samplerate=16000, lowfreq=0, highfreq=None):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq = highfreq or samplerate/2
        assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        lowmel = self._hz_to_mel(lowfreq)
        highmel = self._hz_to_mel(highfreq)
        melpoints = np.linspace(lowmel, highmel, nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((nfft+1)*self._mel_to_hz(melpoints)/samplerate)

        fbank = np.zeros([nfilt, nfft//2])
        for j in range(0, nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j, i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j, i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        return fbank

    def create_mel_filter(self):
        """
        Creates a filter to convolve with the spectrogram to get out mels
        """
        mel_inversion_filter = self.get_filterbanks(nfilt=self.n_freq_components,
                                                    nfft=self.fft_size,
                                                    samplerate=self.samplerate,
                                                    lowfreq=self.start_freq,
                                                    highfreq=self.end_freq
                                                    )
        # Normalize
        mel_filter = mel_inversion_filter.T / mel_inversion_filter.sum(axis=1)
        return mel_filter, mel_inversion_filter

    @property
    def spectrogram(self):
        """Generates training data in the form of spectrogram
        """
        return self.pretty_spectrogram(self.data.astype('float64'))

    @property
    def mel_spectrogram(self):
        """Generates training data in the form of mel spectrogram
        """
        mel_filter, mel_inversion_filter = self.create_mel_filter()
        mel_spec = self.make_mel(self.spectrogram, mel_filter)

        return mel_spec

    def save_images(self, path_to_save):
        """Save mel and spectrogram representations to file for debugging
        Args:
            path_to_save (str) : the path to directory to save img files
        Returns:
            saved filepath locations
        """
        if path_to_save is None:
            raise ValueError('path_to_save must not be None')

        file_suffix = self.filepath.split('/')[-1].split('.')[0]
        spectrogram_path = path_to_save + file_suffix + '_spectrogram.png'
        mel_spectrogram_path = path_to_save + file_suffix + '_mel_spectrogram.png'

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
        cax = ax.matshow(np.transpose(self.spectrogram),
                         interpolation='nearest',
                         aspect='auto',
                         cmap=plt.cm.afmhot,
                         origin='lower'
                         )
        fig.colorbar(cax)
        plt.title('Original Spectrogram')
        plt.savefig(spectrogram_path)

        plt.clf()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
        cax = ax.matshow(self.mel_spectrogram,
                         interpolation='nearest',
                         aspect='auto',
                         cmap=plt.cm.afmhot,
                         origin='lower'
                         )
        fig.colorbar(cax)
        plt.title('mel Spectrogram')
        plt.savefig(mel_spectrogram_path)

        return spectrogram_path, mel_spectrogram_path


#####################################
#             TESTING               #
#####################################


if __name__ == '__main__':

    filepath = '/home/vaz/projects/beat-machine/data/test/wavs/30_132.wav'
    PATH_TO_SAVE = '/home/vaz/projects/beat-machine/data/test/img/'

    filepath = 'F:/spectrogram/sample.wav'
    filepath = 'sample.wav'
    PATH_TO_SAVE = 'F:/spectrogram/img/'

    '''
    data_obj = DataProcessor(filepath=filepath)

    data_obj._load_data

    spec = data_obj.spectrogram

    mel = data_obj.mel_spectrogram

    saved_to = data_obj.save_images(path_to_save=PATH_TO_SAVE)
    

    print(spec)
    print(type(spec))
    print(len(spec))
    print('\n')

    print(mel)
    print(type(mel))
    print(len(mel))
    print('\n')

    print('img saved to: ', saved_to)

    '''

    dp = DataProcessor(filepath = filepath)

    dp.load_data(rate = 1000, n_secs = 3)


    dp.write_to_file_test(dp.rate, dp.data)