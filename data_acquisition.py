"""
This script downloads tracks with valid metadata in batches from captain-hammer.

Raw wav files created are then used with feature_extraction.py
"""
import logging
import sys
from pathlib import Path
from tempfile import TemporaryFile

import requests
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
    """ Downloads metadata for a given track id from
        server and extracts bpm and format data.

    Args: beets_id (int or str) : numeric character correspoding to
                    the track's id on server
    Returns: format (str) : the file format of the track.
             bpm (int) : the int floored bpm of the track
    """
    logger.info('Getting beets track <%s> metadata…', beets_track_url)
    metadata = requests.get(beets_track_url).json()
    bpm = int(metadata.get('bpm', 0)) or None
    genre = metadata.get('genre','')
    file_format = metadata.get('format', '').lower() or None
    return bpm, file_format, genre


def download_beets_track_file(beets_track_url):
    """Downloads the beets track audio and returns a temporary file handle.
    Args: beats_track_url (str) : url of track on server
    Returns: file handle
    """
    logger.info('Downloading beets track <%s> audio file…', beets_track_url)
    f = TemporaryFile()
    f.write(requests.get(beets_track_url + '/file').content)
    f.seek(0)   # in case we want to read this later
    return f


def convert_mp3_to_wav_file(mp3_file):
    """Converts downloaded mp3 to wav."""
    logger.debug('Converting %s to WAV…', mp3_file)
    sound = AudioSegment.from_mp3(mp3_file)
    sound = sound.set_channels(1)
    wav_file = TemporaryFile()
    sound.export(wav_file, format="wav")
    wav_file.seek(0)
    return wav_file


def download_all_beets_tracks():
    """Download all beets tracks with a non-zero BPM.
    """
    # see <http://beets.readthedocs.io/en/v1.4.5/reference/query.html#query-term-negation>
    beets_tracks_with_bpm = requests.get(BEETS_API_ROOT + 'query/^bpm:0').json()['results']
    for track in beets_tracks_with_bpm:
        beets_track_url = BEETS_API_ROOT + str(track['id'])
        bpm = track['bpm']
        file_format = ['format']
        audio_file = download_beets_track_file(beets_track_url)

        if file_format == 'MP3':
            wav_file = convert_mp3_to_wav_file(audio_file)
            audio_file.close()
            audio_file = wav_file   # switcheroo

        TEST_WAVS_DIRECTORY.joinpath('%s.wav', track['id']).write_bytes(audio_file.read())
        audio_file.close()

def downnsample_wav(src, dst, inrate=44100, outrate=16000,
                    inchannels=2, outchannels=1):

    s_read = wave.open(src, 'r')
    s_write = wave.open(dst, 'w')

    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)

    converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
    if outchannels == 1:
        converted = audioop.tomono(converted[0], 2, 1, 0)

    s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
    s_write.writeframes(converted)
    s_read.close()
    s_write.close()


def main(beets_ids):
    """Downloads a batch of tracks from server in WAV format.

    Downloads metadata and checks for bpm tag and format.
    If bpm tag does not exist/is zero, track is not downloaded.
    If track is not in WAV format, convert to it first.

    Args (list[str]) : list of id's to download from server
    """
    for beets_id in beets_ids:
        beets_track_url = BEETS_API_ROOT + str(beets_id)
        bpm, file_format, genre = get_beets_track_bpm_and_format_tags(beets_track_url)
        if not bpm or not file_format:
            logger.warning('No BPM/file format for <%s> (bpm=%s, format=%s).',
                            beets_track_url, bpm, file_format)
            continue
        logger.debug('Beets track <%s> has bpm=%s, format=%s.',
                      beets_track_url, bpm, file_format)
        if file_format == 'mp3':
            mp3_file = download_beets_track_file(beets_track_url)
            wav_file = convert_mp3_to_wav_file(mp3_file)
            mp3_file.close()    # TemporaryFile gets deleted on close
        elif file_format == 'wav':
            wav_file = download_beets_track_file(beets_track_url)

        # TODO: pass file handle directly to next program
        # copy WAV data to TEST_WAVS_DIRECTORY
        TEST_WAVS_DIRECTORY.joinpath('%s_%s_%s.wav' % (genre, beets_id, bpm)).write_bytes(wav_file.read())
        wav_file.close() # remember to close TemporaryFile for deletion


if __name__ == '__main__':
    # download the first 50 tracks!
    main(list(range(1,51)))
