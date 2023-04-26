import numpy as np
import os
import traceback
from glob import glob
import scipy.io.wavfile as siw

def _extract_time_info(info_file_path : str):
    time_info = None
    try:
        with open(info_file_path, "r") as info_if:
            info_data = info_if.readlines()
            time_info = float(info_data[0].split(" ")[-1])
    except Exception as e:
        print(traceback.format_exc())
    finally:
        return time_info

def _create_audio_views(audio_data, window_size, overlap):
    # Calculate the number of samples per step
    step = window_size - overlap
    print("Step: ", step)
    
    # Calculate the number of views
    num_views = int(np.ceil((len(audio_data) - overlap) / step))
    print("Num views", num_views)

    # Create an empty array to store the views
    # audio_views = np.zeros((num_views, window_size))

    audio_views = list()

    # Create views with a sliding window using numpy indexing
    for i in range(num_views):
        start = i * step
        print("START: ", start)
        end = start + window_size
        print("END: ", end)
        audio_views.append(audio_data[start:end])
        print("index: ", i)
        print("View shape: ", audio_views[i].shape)

    # Concatenate the views back into a bigger array
    audio_data_new = np.concatenate(audio_views, axis=0)

    return audio_data_new

audio_files = glob("data/*/*.wav")

for audio_path in audio_files[:1]:
    car_label = audio_path.split('/')[1]
    filename = audio_path.split('/')[-1]

    input_audiofile = siw.read(audio_path)
    sampling_rate, audio = input_audiofile[0], input_audiofile[1]
    print("SR: ", sampling_rate)
    print("Original shape: ", audio.shape)
    file_time_info = int(_extract_time_info(info_file_path = f"{audio_path.split('.')[0]}.txt"))
    view = _create_audio_views(audio_data=np.squeeze(audio[:480000, :1]), window_size=2*sampling_rate, overlap=1*sampling_rate)
    print("View: ", view.shape)
    siw.write(filename=f"{filename}", rate=sampling_rate, data=view)