import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import scipy.io.wavfile as siw

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

    # zero_padding = tf.zeros(sampling_rate - tf.shape(audio), dtype=tf.float32)
    # audio_padded = tf.concat([audio, zero_padding], axis=0)

    return sampling_rate, audio, label

def get_spectrogram(filename):

    frame_length_in_s = 0.04
    frame_step_in_s = 0.02

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

    return stft, spectrogram, sampling_rate_float32, label

def get_log_mel_spectrogram(filename):
    frame_length_in_s = 0.04
    num_mel_bins = 40
    lower_frequency = 20
    upper_frequency = 4000

    stft, spectrogram, sampling_rate_float32, label = get_spectrogram(filename)

    # sampling_rate_float32 = tf.cast(sampling_rate, tf.float32)
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

train_file_ds = tf.data.Dataset.list_files(["complete_training_data/*.wav"])
train_mel_ds = train_file_ds.map(get_log_mel_spectrogram)
for spectrogram, label in train_mel_ds.take(1):
    SHAPE = spectrogram.shape

def preprocess_with_resized_mel(filename):
    log_mel_spectrogram, label = get_log_mel_spectrogram(filename)
    log_mel_spectrogram.set_shape(SHAPE)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :]
    mfccs = tf.expand_dims(mfccs, -1)
    mfccs = tf.image.resize(mfccs, [32, 32])
    label_id = tf.argmax(label == LABELS)

    return mfccs, label_id

# train_file_ds = tf.data.Dataset.list_files(["complete_training_data/*.wav"])
train_ds = train_file_ds.map(preprocess_with_resized_mel)

# TESTING FOR GET_DATA -> ALL OK
# for sampling_rate, audio, label in train_ds:
#     print(sampling_rate, audio.dtype, audio.shape)

# TESTING GET_SPECTROGRAM -> TUTTO FORSE OK
# for stft, spectrogram, sampling_rate, label in train_ds:
#     print(spectrogram.dtype, spectrogram.shape, stft.dtype, stft.shape)

# TESTING GET_LOG_MEL_SPECTROGRAM -> TUTTO FORSE OK
# for log_mel_spectrogram, label in train_ds:
#     print(log_mel_spectrogram.dtype, label)

# TESTING PREPROCESS_WITH_RESIZED_MEL -> 
for mfccs, label_id in train_ds:
    print(mfccs.dtype, mfccs.shape)