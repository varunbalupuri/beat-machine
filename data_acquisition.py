"""
This script downloads tracks with valid metadata in batches from captain-hammer.

Raw wav files created are then used with feature_extraction.py
"""
import logging
from pathlib import Path
from tempfile import TemporaryFile

import requests
import sys
from pydub import AudioSegment


# logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_log_handler = logging.StreamHandler(sys.stdout)
stdout_log_handler.setLevel(logging.DEBUG)
logger.addHandler(stdout_log_handler)

BEETS_API_ROOT = 'http://beets.westerndecadence.com/item/'
TEST_WAVS_DIRECTORY = Path.cwd().joinpath('data/test/wavs/')


def get_beets_track_bpm_and_format_tags(beets_track_url):
    """ Downloads metadata for a given track id from server and extracts bpm and format data.

    Args: beets_id (int or str) : numeric character correspoding to the track's id on server
    Returns: format (str) : the file format of the track.
             bpm (int) : the int floored bpm of the track
    """
    logger.info('Getting beets track <%s> metadata…', beets_track_url)
    metadata = requests.get(beets_track_url).json()
    bpm = int(metadata.get('bpm', 0)) or None
    file_format = metadata.get('format', '').lower() or None
    return bpm, file_format


def download_beets_track_file(beets_track_url):
    """Downloads the beets track audio and returns a temporary file handle.
    """
    logger.info('Downloading beets track <%s> audio file…', beets_track_url)
    f = TemporaryFile()
    f.write(requests.get(beets_track_url + '/file').content)
    f.seek(0)   # in case we want to read this later
    return f


def convert_mp3_to_wav_file(mp3_file):
    """Converts downloaded mp3 to wav and deletes mp3."""
    logger.debug('Converting %s to WAV…', mp3_file)
    sound = AudioSegment.from_mp3(mp3_file)
    wav_file = TemporaryFile()
    sound.export(wav_file, format="wav")
    wav_file.seek(0)
    return wav_file


def main(beets_ids):
    """Downloads a batch of tracks from server in WAV format.

    Downloads metadata and checks for bpm tag and format.
    If bpm tag does not exist/is zero, track is not downloaded.
    If track is not in WAV format, convert to it first.

    Args (list[str]) : list of id's to download from server
    """
    for beets_id in beets_ids:
        beets_track_url = BEETS_API_ROOT + str(beets_id)
        bpm, file_format = get_beets_track_bpm_and_format_tags(beets_track_url)
        if not bpm or not file_format:
            logger.warning('No BPM/file format for <%s> (bpm=%s, format=%s).', beets_track_url, bpm, file_format)
            continue
        logger.debug('Beets track <%s> has bpm=%s, format=%s.', beets_track_url, bpm, file_format)
        if file_format == 'mp3':
            mp3_file = download_beets_track_file(beets_track_url)
            wav_file = convert_mp3_to_wav_file(mp3_file)
            mp3_file.close()    # TemporaryFile gets deleted on close
        elif file_format == 'wav':
            wav_file = download_beets_track_file(beets_track_url)

        # TODO: pass file handle directly to next program
        # copy WAV data to TEST_WAVS_DIRECTORY
        TEST_WAVS_DIRECTORY.joinpath('%s.wav' % beets_id).write_bytes(wav_file.read())
        wav_file.close()    # remember to close TemporaryFile for deletion

'''
def wav_to_array(directory, Xdim):
    """extract vector of length Xdim from wav file.
    """
    Xs = []
    Ys = [124, 125, 122]

    for filepath in os.listdir(directory):
        print(directory+filepath)

        sample_rate, data = wavfile.read(directory+filepath)
        #mono = data.mean(axis=1)

        print(sample_rate, data)

        data = np.array(data[10000:10000+Xdim]).astype(np.float32)
        data = normalise(data)
        Xs.append(data)

    return Xs, Ys
'''

if __name__ == '__main__':
    main([1, 2, 3, 4, 5])
