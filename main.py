from preprocessing import DatasetFormatter, get_audio_and_label

df = DatasetFormatter(crop_time=2)
df.format_dataset()

# get_audio_and_label("formatted_data/formatted_CitroenC4Picasso/CitroenC4Picasso_35.wav")

# import tensorflow as tf

# def read_audio(filename):
#     audio_binary = tf.io.read_file(filename)
#     audio, sampling_rate = tf.audio.decode_wav(audio_binary)
#     print(audio.shape)
#     print(sampling_rate)
    

# read_audio(filename="down_2d82a556_nohash_1.wav")