# dataset_generator.py

""" Constructs a compliant dataset with 
"""
import logging
import datetime
import sys
import os
import glob
import pickle
import logging
import numpy as np

from data_processor import DataProcessor

from data_acquisition import (get_beets_track_bpm_and_format_tags,
                              genre_is_of_interest,
                              bpm_is_in_range,
                              get_beets_track_bpm_and_format_tags,
                              download_beets_track_file,
                              convert_mp3_to_wav_file,
                              BEETS_API_ROOT,
                              TEST_WAVS_DIRECTORY)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_log_handler = logging.StreamHandler(sys.stdout)
stdout_log_handler.setLevel(logging.DEBUG)
logger.addHandler(stdout_log_handler)


N_SECS = 3
MEL = True
CHUNK_SIZE = 7392



LOWER_BOUND = 123
UPPER_BOUND = 135


PATH_TO_WAVS = os.getcwd() + '/data/test/wavs/'


# meta is a dict {'beets_id': int, 'genre': int}
dataset = {'X': [], 'y': [], 'meta': []}


def get_metadata(filepath):
    """Gets bpm, genre and beets_id of file from filename.
    Args:
        filepath (str): file location of raw wav file

    Returns:
        genre (str): genre of the track
        beets_id (int): the beets id of the track.
        bpm (int): bpm of track
    """

    filename = filepath.split('/')[-1]
    root = filename.split('.wav')[0].split('_')

    genre = root[0]
    beets_id = root[1]
    bpm = int(root[-1])

    return genre, beets_id, bpm


def load_tracks_into_dataset(dataset):
    """ Loads all valid tracks in PATH_TO_WAVS
    into existing dataset

    Args:
        dataset (dict): Description
    """
    files = glob.glob(PATH_TO_WAVS+'*.wav')
    for file in files:
        try:
            genre, beets_id, y = get_metadata(file)

            for i in range(15):
                dataset['y'].append(np.array([y]))

                dp = DataProcessor(filepath=file)
                dp.load_data(n_secs=N_SECS)

                if MEL:
                    X = dp.mel_spectrogram
                else:
                    X = dp.spectrogram

                if X.shape != (64, 1022):
                    continue

                dataset['X'].append(X.astype(float))
                meta = {'beets_id': beets_id, 'genre': genre}

                dataset['meta'].append(meta)
        except Exception:
            logger.warning('Error loading id: {}'.format(beets_id), exc_info=True)


def download_chunks(start, chunksize,
                    error_counter=0, got_counter=0, total_counter=0):

    for beets_id in range(start, start+chunksize):
        try:
            total_counter += 1
            beets_track_url = BEETS_API_ROOT + str(beets_id)
            bpm, file_format, genre = get_beets_track_bpm_and_format_tags(beets_track_url)
            if not bpm or not file_format:
                logger.warning('No BPM/file format for <%s> (bpm=%s, format=%s).',
                               beets_track_url, bpm, file_format)
                continue

            if not genre_is_of_interest(genre):
                continue

            if not bpm_is_in_range(bpm, LOWER_BOUND, UPPER_BOUND):
                continue

            logger.debug('Beets track <%s> has bpm=%s, format=%s.',
                         beets_track_url, bpm, file_format)

            if file_format == 'mp3':
                mp3_file = download_beets_track_file(beets_track_url)
                wav_file = convert_mp3_to_wav_file(mp3_file)
                mp3_file.close()   # TemporaryFile gets deleted on close
                got_counter += 1

            elif file_format == 'wav':
                wav_file = download_beets_track_file(beets_track_url)
                got_counter += 1

            TEST_WAVS_DIRECTORY.joinpath('%s_%s_%s.wav' % (genre, beets_id, bpm)).write_bytes(wav_file.read())
            wav_file.close()  # remember to close TemporaryFile for deletion

            logger.info('scanned {} tracks: got {}'.format(total_counter, got_counter))
        except Exception as e:
            logger.warning('Error with id {}'.format(beets_id), exc_info=True)
            error_counter += 1

    print('error_counter', error_counter)
    print('total_counter', total_counter)
    print('got_counter', got_counter)


def clear_data_directory():
    files = files = glob.glob(PATH_TO_WAVS+'*.wav')
    for f in files:
        os.remove(f)


def pickle_dataset(dtatwaset):
    if MEL:
        spectrogram_type = 'mel'
    else:
        spectrogram_type = 'spec'

    date = str(datetime.datetime.now().date())
    len_X = len(dataset['X'])
    dataset_save_name = '{}_dataset_{}_len_{}.pickle'.format(spectrogram_type,
                                                             date, len_X)

    with open('datasets/{}'.format(dataset_save_name),
              'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    download_chunks(1, CHUNK_SIZE)
    load_tracks_into_dataset(dataset)

    clear_data_directory()

if __name__ == '__main__':
    main()

    pickle_dataset(dataset)
