# mel spectrogram working implementation

import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage

# Adapted from: https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe

"""
TO DO:
> document params
> sort out feeding args as self.xxxx
> manual garbage collection of data - lots of memory used per object :(
> add unit tests
> README
> 
"""

class DataProcessor(object):
    """
    Contains the methods required for turning
    raw audio data into spectrogram arrays
    for use in neural network. Also contains
    data visualisation and introspection
    methods
    """

    def __init__(self, filepath, dir_to_save_to, lowcut=500, highcut=15000,
                 fft_size=2048, spec_thresh=4, n_mel_freq_components=64,
                shorten_factor = 10, start_freq = 300, end_freq = 8000 ):
        """

        Parameters

        fft_size = 2048 # window size for the FFT
        step_size = fft_size/16 # distance to slide along the window (in time)
        spec_thresh = 4 # threshold for spectrograms (lower filters out more noise)
        lowcut = 500 # Hz # Low cut for our butter bandpass filter
        highcut = 15000 # Hz # High cut for our butter bandpass filter
        #### For mels ###
        n_mel_freq_components = 64 # number of mel frequency channels
        shorten_factor = 10 # how much should we compress the x-axis (time)
        start_freq = 300 # Hz # What frequency to start sampling our melS from 
        end_freq = 8000 # Hz # What frequency to stop sampling our melS from 

        """
        self.filepath = filepath
        self.lowcut = lowcut
        self.highcut = highcut
        self.rate, self.data = wavfile.read(filepath)
        #TODO: Add all these vars into init

    def _hz2mel(self, hz):
        """Convert a value in Hertz to Mels
        :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 2595 * np.log10(1+hz/700.)
        
    def _mel2hz(self, mel):
        """Convert a value in Mels to Hertz
        :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        return 700*(10**(mel/2595.0)-1)


    def _butter_bandpass(self, fs, order=5):
        nyqist_freq = 0.5 * fs
        low =  self.lowcut / nyqist_freq
        high = self.highcut / nyqist_freq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, fs, order=5):
        b, a = self._butter_bandpass(fs, order=order)
        y = lfilter(b, a, data)
        return y

    def overlap(self, X, window_size, window_step):
        """
        Create an overlapped version of X
        Parameters
        ----------
        X : ndarray, shape=(n_samples,)
            Input signal to window and overlap
        window_size : int
            Size of windows to take
        window_step : int
            Step size between windows
        Returns
        -------
        X_strided : shape=(n_windows, window_size)
            2D array of overlapped X
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
        out = np.ndarray((nw,ws),dtype = a.dtype)

        for i in range(nw):
            # "slide" the window along the samples
            start = int(i * ss)
            stop = int(start + ws)
            out[i] = a[start : stop]

        return out

    def stft(self, X, fftsize=128, step=65, mean_normalize=True, real=False,
             compute_onesided=True):
        """
        Compute STFT for 1D real valued input X
        """
        if real:
            local_fft = np.fft.rfft
            cut = -1
        else:
            local_fft = np.fft.fft
            cut = None
        if compute_onesided:
            cut = int(fftsize / 2)
        if mean_normalize:
            X -= X.mean()

        X = overlap(X, fftsize, step)
        
        size = fftsize
        win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
        X = X * win[None]
        X = local_fft(X)[:, :cut]
        return X

    def pretty_spectrogram(self, d, log = True, thresh= 5, fft_size = 512, step_size = 64):
        """
        creates a spectrogram
        log: take the log of the spectrgram
        thresh: threshold minimum power for log spectrogram
        """
        specgram = np.abs(self.stft(d, fftsize=fft_size, step=step_size, real=False, compute_onesided=True))
      
        if log == True:
            specgram /= specgram.max() # volume normalize to max 1
            specgram = np.log10(specgram) # take log
            specgram[specgram < -thresh] = -thresh # set anything less than the threshold as the threshold
        else:
            specgram[specgram < thresh] = thresh # set anything less than the threshold as the threshold
        
        return specgram

    def make_mel(self, spectrogram, mel_filter, shorten_factor = 1):
        mel_spec =np.transpose(mel_filter).dot(np.transpose(spectrogram))
        mel_spec = scipy.ndimage.zoom(mel_spec.astype('float32'), [1, 1./shorten_factor])#.astype('float16')
        mel_spec = mel_spec[:,1:-1] # a little hacky but seemingly needed for clipping 
        return mel_spec


    def get_filterbanks(self, nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq= highfreq or samplerate/2
        assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
        
        # compute points evenly spaced in mels
        lowmel = self._hz2mel(lowfreq)
        highmel = self._hz2mel(highfreq)
        melpoints = np.linspace(lowmel,highmel,nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((nfft+1)*self._mel2hz(melpoints)/samplerate)

        fbank = np.zeros([nfilt,nfft//2])
        for j in range(0,nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        return fbank

    def create_mel_filter(self, fft_size, n_freq_components = 64, start_freq = 300, end_freq = 8000, samplerate=44100):
        """
        Creates a filter to convolve with the spectrogram to get out mels

        """
        mel_inversion_filter = get_filterbanks(nfilt=n_freq_components, 
                                               nfft=fft_size, samplerate=samplerate, 
                                               lowfreq=start_freq, highfreq=end_freq)
        # Normalize filter
        mel_filter = mel_inversion_filter.T / mel_inversion_filter.sum(axis=1)

        return mel_filter, mel_inversion_filter

    @property
    def spectrogram(self):
        return self.pretty_spectrogram(self.data.astype('float64'), fft_size = fft_size, 
                                   step_size = step_size, log = True, thresh = spec_thresh)


    @property
    def mel_spectrogram(self):
        mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq)

        mel_spec = make_mel(wav_spectrogram, mel_filter, shorten_factor = shorten_factor)


    def save_images(self, path_to_save)
        #save mel and spectrogram representations to file for debugging

        spectrogram_path = path_to_save+'_spectrogram.png'
        mel_spectrogram_path =path_to_save+'_mel_spectrogram.png'

        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
        # TO DO throw error if spec not properly initialised
        cax = ax.matshow(np.transpose(self.spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
        fig.colorbar(cax)
        plt.title('Original Spectrogram')
        plt.savefig(path_to_save+'_spectrogram.png')

        plt.clf()

        # plot the compressed spec
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))

        cax = ax.matshow(mel_spec, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
        fig.colorbar(cax)
        plt.title('mel Spectrogram')
        plt.savefig(path_to_save+'_spectrogram.png')

        return spectrogram_path, mel_spectrogram_path


#####################################
#             TESTING               #
#####################################


if __name__ == '__main__':
	### Parameters ###
	fft_size = 2048 # window size for the FFT
	step_size = fft_size/16 # distance to slide along the window (in time)
	spec_thresh = 4 # threshold for spectrograms (lower filters out more noise)
	lowcut = 500 # Hz # Low cut for our butter bandpass filter
	highcut = 15000 # Hz # High cut for our butter bandpass filter
	# For mels
	n_mel_freq_components = 64 # number of mel frequency channels
	shorten_factor = 10 # how much should we compress the x-axis (time)
	start_freq = 300 # Hz # What frequency to start sampling our melS from 
	end_freq = 8000 # Hz # What frequency to stop sampling our melS from 

	# Grab your wav and filter it
	mywav = '/home/vaz/projects/beat-machine/data/test/wavs/30_132.wav'
	rate, data = wavfile.read(mywav)
	data = butter_bandpass_filter(data, lowcut, highcut, rate, order=1)
	# Only use a short clip for our demo
	if np.shape(data)[0]/float(rate) > 10:
	    data = data[0:rate*10] 
	print('Length in time (s): ', np.shape(data)[0]/float(rate))


	wav_spectrogram = pretty_spectrogram(data.astype('float64'), fft_size = fft_size, 
                                   step_size = step_size, log = True, thresh = spec_thresh)

    print(wav_spectrogram)
    print(type(wav_spectrogram))

	fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
	cax = ax.matshow(np.transpose(wav_spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
	fig.colorbar(cax)
	plt.title('Original Spectrogram')

	#plt.savefig('spec_vs_mel/'+filename+'.png')

	plt.show()

	plt.clf()

	mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq)

	mel_spec = make_mel(wav_spectrogram, mel_filter, shorten_factor = shorten_factor)


    print(mel_spec)
    print(type(mel_spec))

	# plot the compressed spec
	fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))

	cax = ax.matshow(mel_spec, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
	fig.colorbar(cax)
	plt.title('mel Spectrogram')

	plt.show()