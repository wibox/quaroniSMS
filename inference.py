# deve poter registrare un file audio
# questo file audio deve essere processato according to preprocessing
# deve essere classificato tramite la rete trainata
# e deve essere inserito all'interno di redis
import redis
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import pandas as pd
import scipy.io.wavfile as siw
import sounddevice as sd

from typing import *
from datetime import datetime
from glob import glob
from traceback import *
import time
import argparse
import psutil
import uuid
import os

def safe_ts_create(redis_client, key):
    try:
        redis_client.ts().create(key)
    except redis.ResponseError:
        pass

def get_audio_from_nparray(input_data : np.ndarray = None) -> tf.Tensor:
    try:
        if input_data:
            input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
            input_data = 2*((input_data+32768)/(32767+32768))-1
            input_data = tf.squeeze(input_data)
            return input_data
        else:
            raise Exception("NoneType or Empty nparray provided.")
    except TypeError as te:
        print(te.format_exc())
    except Exception as e:
        print(e.format_exc())
    finally:
        return input_data

def get_spectrogram(input_data : np.ndarray, downsampling_rate : int, frame_length_in_s : float, frame_step_in_s : float, sampling_rate : int) -> Tuple[Any, int]:
    audio_padded = get_audio_from_nparray(input_data=input_data)

    if sampling_rate != downsampling_rate:
        sampling_rate_int64 = tf.cast(sampling_rate, tf.int64)
        audio_padded = tfio.audio.resample(audio_padded, sampling_rate_int64, downsampling_rate)
     
    sampling_rate_float32 = tf.cast(sampling_rate, tf.float32)
    frame_length = int(frame_length_in_s*sampling_rate_float32)
    frame_step = int(frame_step_in_s*sampling_rate_float32)

    stft = tf.signal.stft(
        audio_padded,
        frame_length = frame_length,
        frame_step = frame_step,
        fft_length = frame_length
    )

    spectrogram = tf.abs(stft)

    return spectrogram, sampling_rate

def get_ltmwm(num_mel_bins : int, num_spectrogram_bins : int, downsampling_rate : int, lower_frequency : int, upper_frequency : int) -> Any:
    return tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, downsampling_rate, lower_frequency, upper_frequency)

def get_mfccs_features(spectrogram : Any, linear_to_mel_weight_matrix : Any, how_many : int = 13) -> tf.Tensor:
    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms=log_mel_spectrogram)
    mfccs = mfccs[..., :how_many]
    mfccs = tf.expand_dims(mfccs, 0)
    mfccs = tf.expand_dims(mfccs, -1)
    mfccs = tf.image.resize(mfccs, [32, 32])

    return mfccs

def is_silence(input_data : np.ndarray, downsampling_rate : int, frame_length_in_s : float,  dbFSthresh : float, sampling_rate : int, duration_time : float):
    spectrogram, _ = get_spectrogram(
        input_data=input_data,
        downsampling_rate=downsampling_rate,
        frame_length_in_s=frame_length_in_s,
        frame_step_in_s=frame_length_in_s,
        sampling_rate=sampling_rate
    )

    dbFS = 20 * tf.math.log(spectrogram + 1.e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)
    non_silence = energy > dbFSthresh
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1) * frame_length_in_s

    if non_silence_duration > duration_time:
        return 0
    else:
        return 1

def callback(indata, frames, callback_time, status):
        mydata = get_audio_from_nparray(input_data=indata)
        if not is_silence(input_data=mydata, downsampling_rate=48000, frame_length_in_s=0.04, dbFSthresh=-100, duration_time=0.01, sampling_rate=48000):
            timestamp = time.time()
            siw.write(filename=f"recordings/{timestamp}.wav", rate=48000, data=indata)

def resample_audio_array(audio_padded : Any, downsampling_rate : int) -> Any:
    return tfio.audio.resample(audio_padded, tf.cast(downsampling_rate, tf.int64), downsampling_rate)

def speed_classification(
    filename : str,
    downsampling_rate : int,
    frame_length_in_s : float,
    frame_step_in_s : float,
    num_mfccs_features: int,
    linear_to_mel_weight_matrix : Any,
    interpreter : tf.lite.Interpreter,
    input_details : List[Dict[str, Any]],
    output_details : List[Dict[str, Any]],
    labels : List[str]
) -> Union[str, None]:

    audio_binary = siw.read(filename)
    sampling_rate, audio = audio_binary[0], audio_binary[1]
    audio = tf.squeeze(audio)
    # zero_padding = tf.zeros(sampling_rate - tf.shape(audio), dtype=tf.float32)
    # audio_padded = tf.concat([audio, zero_padding], axis=0)

    # if sampling_rate != downsampling_rate:
    #     audio_padded = resample_audio_array(audio_padded=audio_padded, downsampling_rate=downsampling_rate)

    spectrogram, _ = get_spectrogram(
        input_data=audio,
        downsampling_rate=downsampling_rate,
        frame_length_in_s=frame_length_in_s,
        frame_step_in_s=frame_step_in_s,
        sampling_rate=sampling_rate
    )

    mfccs_features = get_mfccs_features(spectrogram=spectrogram, linear_to_mel_weight_matrix=linear_to_mel_weight_matrix, how_many=num_mfccs_features)
    interpreter.set_tensor(input_details[0]['index'], mfccs_features)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    if np.max(output[0]) > .95:
        os.remove(f"{filename}")
        return labels[np.argmax(output[0])]
    else:
        os.remove(f"{filename}")
        return None

def redis_connection(args) -> Tuple[bool, redis.Redis]:
    print("Connecting to database...")
    redis_client = redis.Redis(
        host = args.host, 
        password = args.password, 
        username = args.user, 
        port = args.port
        )
    result = redis_client.ping()
    print("Is connected? ", result)
    return result, redis_client

def load_task_details(model_name : str):
    interpreter = tf.lite.Interpreter(model_path=f"tflite_model/{model_name}.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details

def load_parameters_from_csv(filepath : str) -> bool:
    try:
        df = pd.read_csv(filepath)
        # downsampling_rate = int(df["downsampling_rate"])
        downsampling_rate = 48000
        frame_length_in_s = float(df["frame_length_in_s"])
        frame_step_in_s = float(df["frame_step_in_s"])
        num_mel_bins = int(df["num_mel_bins"])
        lower_frequency = float(df["lower_frequency"])
        upper_frequency = float(df["upper_frequency"])
        num_mfccs_features = int(df["num_mfccs_features"])
    except Exception:
        print("spectrogram_results.csv not found. Please download it from the provided notebook on DeepNote or run the provided training.ipynb.")
    finally:
        return downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency, num_mfccs_features

def update_informations(predicted_label : str) -> Tuple[float, float, bool]:
    ts_in_s = time.time()
    ts_in_ms = int(ts_in_s*1000)
    mac_id = hex(uuid.getnode())
    formatted_datetime = datetime.fromtimestamp(ts_in_s)
    print(f"{formatted_datetime} - {mac_id}: speed_label ", predicted_label)

    return ts_in_ms, predicted_label

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type = str, default = "redis-15072.c77.eu-west-1-1.ec2.cloud.redislabs.com")
    parser.add_argument("--port", type = int, default = 15072)
    parser.add_argument("--user", type = str, default = "default")
    parser.add_argument("--password", type = str, default = "53R8YAlL81zAHIEVcPjwjzcnVQoSPhzt")
    parser.add_argument("--device", type = int, default = 1)
    parser.add_argument("--model-name", type=str, default="1678110341")
    parser.add_argument("--csv-path", type=str, default="hw2_log_final.csv")
    args = parser.parse_args()
    
    _, redis_client = redis_connection(args=args)
    
    safe_ts_create(redis_client, "mac_adress:speed")

    interpreter, input_details, output_details = load_task_details(model_name=args.model_name)
    downsamplig_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency, num_mfccs_features= load_parameters_from_csv(filepath=args.csv_path)
    num_spectrogram_bins = int(int(frame_length_in_s*tf.cast(downsamplig_rate, tf.float32)) // 2 + 1)
    linear_to_mel_weight_matrix = get_ltmwm(num_mel_bins=num_mel_bins, num_spectrogram_bins=num_spectrogram_bins, downsampling_rate=downsamplig_rate, lower_frequency=lower_frequency, upper_frequency=upper_frequency)

    info_monitoring = False

    if not os.path.exists("recordings/"):
        os.makedirs("recordings")

    with sd.InputStream(device=args.device, channels=1, samplerate=16000, dtype="float32", callback=callback, blocksize=16000):
        print("Recording audio...")
        while True:
            ipt = input()
            if ipt.lower() == "q":
                print("Processing audio files...")
                for audiofile in glob("recordings/*.wav"):
                    predicted_label = speed_classification(
                        filename=audiofile,
                        downsampling_rate=downsamplig_rate,
                        frame_length_in_s=frame_length_in_s,
                        frame_step_in_s=frame_step_in_s,
                        num_mfccs_features = num_mfccs_features,
                        linear_to_mel_weight_matrix=linear_to_mel_weight_matrix,
                        interpreter=interpreter,
                        input_details=input_details,
                        output_details=output_details,
                        labels=["low", "medium", "high"]
                    )
                    print(predicted_label)

                    if info_monitoring:
                        ts_in_ms, speed_label = update_informations(predicted_label)
                        redis_client.ts().add("mac_adress:speed", ts_in_ms, speed_label)
                time.sleep(1)
            elif ipt.lower() == "p":
                print("Processing audio files stopped.")
                break

if __name__ == '__main__':
    main()