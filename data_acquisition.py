import logging

import requests
import os
import io
import json

from pydub import AudioSegment

logger = logging.getLogger(__name__)


############################################
#  Grab and save the song from cpt-hammer  #
############################################

"""
This script downloads tracks with valid metadata in batches

Raw wav files created are then used with feature_extraction.py
and deleted to conserve disk space.

"""

# CHANGE ME!!
MP3_FOLDER = '/home/vaz/projects/BeePeeM/test_data/beets/mp3s/'
WAV_FOLDER = '/home/vaz/projects/BeePeeM/test_data/beets/wavs/'
BEETS_API_ROOT = 'http://beets.westerndecadence.com/item/'


def get_metadata_tags(beets_id):
    """ Downloads metadata for a given track id from server and extracts bpm and format data.

    Args: beets_id (int or str) : numeric character correspoding to the track's id on server
    Returns: format (str) : the file format of the track.
             bpm (int) : the int floored bpm of the track
    """
    beets_id = str(beets_id)
    r = requests.get(url=BEETS_API_ROOT + beets_id)
    metadata = json.loads(r.content)
    bpm = metadata.get('bpm', 0)
    file_format = metadata.get('format', 0)
    if not file_format:
        print('no valid file format found for id: ' + beets_id)
        return None
    if not bpm:
        print('no valid bpm found for id: ' + beets_id)
        return None
    return int(bpm), file_format.lower()


def beets_save_mp3(beets_id, path_to_save_to=MP3_FOLDER):
    """Downloads file id from server and saves to path_to_save_to.

    Args: beets_if (int or str) : numeric character correspoding to the track's id on server
    Returns path_to_save_to (str) : full path of downloaded mp3 file.
    """
    beets_id = str(beets_id)
    path_to_save_to = path_to_save_to + beets_id + '.mp3'
    song_url = BEETS_API_ROOT + beets_id + '/file'
    a = requests.get(song_url)
    bytes_data = a.content
    with open(path_to_save_to, 'wb') as f:
        f.write(bytes_data)
    print('beets track id: ' + beets_id + ' saved MP3 to : ' + path_to_save_to)
    return path_to_save_to


def beets_save_wav(beets_id, path_to_save_to=WAV_FOLDER):
    beets_id = str(beets_id)
    path_to_save_to = path_to_save_to + beets_id + '.wav'
    song_url = BEETS_API_ROOT + beets_id + '/file'
    a = requests.get(song_url)
    bytes_data = a.content
    with open(path_to_save_to, 'wb') as f:
        f.write(bytes_data)
    print('beets track id: ' + beets_id + ' saved WAV to : ' + path_to_save_to)
    return path_to_save_to


def mp3_to_wav(song_id, mp3_path, wav_folder=WAV_FOLDER):
    """Converts downloaded mp3 to wav and deletes mp3."""
    song_id = str(song_id)
    mp3_data = open(mp3_path, 'rb').read()
    sound = AudioSegment.from_mp3(io.BytesIO(mp3_data))
    path_to_save = WAV_FOLDER + song_id + '.wav'
    sound.export(path_to_save, format="wav")
    # remove the mp3 file
    os.remove(mp3_path)
    print('Converted id: ' + song_id + ' to WAV, saved at : ' + path_to_save)


def main(id_list):
    """Downloads a batch of tracks from server.

    If the track is in wav format, directly save, if it is mp3, convert and save.
        
    Downloads metadata and checks for bpm tag and format.
    If bpm tag does not exist/is zero, track is not downloaded.

    Args (list[str]) : list of id's to download from server
    """
    for beets_id in id_list:
        tags = get_metadata_tags(beets_id)
        if tags is None:
            continue
        bpm, file_format = tags
        print(bpm, file_format)
        if file_format == 'mp3':
            mp3_path = beets_save_mp3(beets_id)
            mp3_to_wav(beets_id, mp3_path)
        elif file_format == 'wav':
            beets_save_wav(beets_id)


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
