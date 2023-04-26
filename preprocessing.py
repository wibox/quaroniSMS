import scipy.io.wavfile as siw

import os
from glob import glob
from tqdm import tqdm

from typing import *
from traceback import *

import tensorflow as tf

import numpy as np

import traceback

LABELS = ["low", "medium", "high"]

def tf_read(filename):
    return tf.numpy_function(siw.read, [filename], [tf.int64, tf.float32])

def get_data(filename):

    sampling_rate, audio = tf_read(filename)
    sampling_rate = tf.cast(sampling_rate, tf.int32)
    audio = tf.cast(audio, dtype=tf.float32)
    audio = (audio + 32768) / (32767 + 32768)
    audio = audio[:, :1]

    path_parts = tf.strings.split(filename, '/')
    path_end = path_parts[-1]
    file_parts = tf.strings.split(path_end, '_')
    label_parts = tf.strings.split(file_parts[-1], ".")
    label = label_parts[0]

    audio = tf.squeeze(audio)

    return sampling_rate, audio, label

def get_spectrogram(filename, frame_step_in_s, frame_length_in_s):

    sampling_rate, audio, label = get_data(filename)

    sampling_rate_float32 = tf.cast(sampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_step = int(frame_step_in_s * sampling_rate_float32)

    stft = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)

    return spectrogram, sampling_rate_float32, label

def get_log_mel_spectrogram(filename, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency):
    # frame_length_in_s = 0.04
    # num_mel_bins = 40
    # lower_frequency = 20
    # upper_frequency = 4000

    spectrogram, sampling_rate_float32, label = get_spectrogram(filename, frame_step_in_s=frame_step_in_s, frame_length_in_s=frame_length_in_s)

    frame_length = int(frame_length_in_s * sampling_rate_float32)
    num_spectrogram_bins = frame_length // 2 + 1

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sampling_rate_float32,
        lower_edge_hertz=lower_frequency,
        upper_edge_hertz=upper_frequency
    )

    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

    label = tf.strings.to_number(label, tf.int32)

    if label <= 40:
        label = "low"
    elif label >= 70:
        label = "high"
    else:
        label = "medium"

    return log_mel_spectrogram, label

class DatasetFormatter():
    def __init__(
        self,
        crop_time : int = 1, 
        window_length : int = 3,
        overlap_size : int = 2,
        standard_audio_limit : int = 480000
    ):

        self.crop_time = crop_time
        self.window_length = window_length
        self.overlap_size = overlap_size
        self.standard_audio_limit = standard_audio_limit
        self.audio_files : List[str] = list()

    def _create_audio_views(self, audio_data, window_size, overlap):
        # Calculate the number of samples per step
        step = window_size - overlap
        
        # Calculate the number of views
        num_views = int(np.ceil((len(audio_data) - overlap) / step))

        # Create an empty array to store the views
        audio_views = list()

        # Create views with a sliding window using numpy indexing
        for i in range(num_views):
            start = i * step
            end = start + window_size
            audio_views.append(audio_data[start:end])

        # Concatenate the views back into a bigger array
        audio_data_new = np.concatenate(audio_views, axis=0)

        return audio_data_new

    def _initialize(self):
        print("Loading audio files informations...")
        self.audio_files = glob("data/*/*.wav")
        print(f"Found {len(self.audio_files)} audio files.")

    def _extract_time_info(self, info_file_path : str):
        time_info = None
        try:
            with open(info_file_path, "r") as info_if:
                info_data = info_if.readlines()
                time_info = float(info_data[0].split(" ")[-1])
        except Exception as e:
            print(traceback.format_exc())
        finally:
            return time_info

    def _crop_audio(self, audio_path : str):

        car_label = audio_path.split('/')[1]
        filename = audio_path.split('/')[-1]

        if not os.path.exists(f"formatted_data/formatted_{audio_path.split('/')[1]}"):
            os.makedirs(f"formatted_data/formatted_{car_label}")

        input_audiofile = siw.read(audio_path)
        sampling_rate, audio = input_audiofile[0], input_audiofile[1]
        file_time_info = int(self._extract_time_info(info_file_path = f"{audio_path.split('.')[0]}.txt"))
        
        # fixing the audio length to standard size with just one channel
        audio = audio[:self.standard_audio_limit, :1]

        if self.crop_time != -1:
            formatted_audio = audio[file_time_info*sampling_rate-sampling_rate*self.crop_time:file_time_info*sampling_rate+sampling_rate*self.crop_time, :]
        else:
            formatted_audio = audio

        siw.write(filename=f"formatted_data/formatted_{car_label}/{filename}", rate=sampling_rate, data=formatted_audio)

    def _window_audio(self, audio_path : str):

        car_label = audio_path.split('/')[1]
        filename = audio_path.split('/')[-1]

        if not os.path.exists(f"formatted_data/formatted_{audio_path.split('/')[1]}"):
            os.makedirs(f"formatted_data/formatted_{car_label}")

        input_audiofile = siw.read(audio_path)
        sampling_rate, audio = input_audiofile[0], input_audiofile[1]

        wl = self.window_length*sampling_rate
        overlap = self.overlap_size*sampling_rate

        formatted_audio = self._create_audio_views(audio_data=audio[:self.standard_audio_limit, :1], window_size=wl, overlap=overlap)

        siw.write(filename=f"formatted_data/formatted_{car_label}/{filename}", rate=sampling_rate, data=formatted_audio)
    
    def _crop_and_window_audio(self, audio_path : str):

        car_label = audio_path.split('/')[1]
        filename = audio_path.split('/')[-1]

        if not os.path.exists(f"formatted_data/formatted_{audio_path.split('/')[1]}"):
            os.makedirs(f"formatted_data/formatted_{car_label}")

        input_audiofile = siw.read(audio_path)
        sampling_rate, audio = input_audiofile[0], input_audiofile[1]
        file_time_info = int(self._extract_time_info(info_file_path = f"{audio_path.split('.')[0]}.txt"))

        audio = audio[:self.standard_audio_limit, :1]

        if self.crop_time != -1:
            formatted_audio = audio[file_time_info*sampling_rate-sampling_rate*self.crop_time:file_time_info*sampling_rate+sampling_rate*self.crop_time, :]
        else:
            formatted_audio = audio

        wl = self.window_length*sampling_rate
        overlap = self.overlap_size*sampling_rate
        
        formatted_audio = self._create_audio_views(audio_data=audio[:, :], window_size=wl, overlap=overlap)
        
        siw.write(filename=f"formatted_data/formatted_{car_label}/{filename}", rate=sampling_rate, data=formatted_audio)

    def format_dataset(self, strategy : str):
        self._initialize()
        if self.crop_time != -1:
            print(f"Cropping audio files to {self.crop_time} seconds before and after passing time. Saving to ./formatted_data/")
        else:
            print("Maintaining original audio files. Saving to ./formatted_data/")
        for audio_path in tqdm(self.audio_files):
            try:
                if strategy == "crop":
                    self._crop_audio(audio_path = audio_path)
                elif strategy == "window":
                    self._window_audio(audio_path = audio_path)
                elif strategy == "CropAndWindow":
                    self._crop_and_window_audio(audio_path = audio_path)
                else:
                    print("Please select one among 'crop', 'window' and 'CropAndWindow' ")
            except Exception as e:
                print(f"Problems reading audio: {audio_path}, skipping it.")
                print(traceback.format_exc())