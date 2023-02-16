from preprocessing import DatasetFormatter, get_audio_and_label

df = DatasetFormatter(crop_time=2)
df.format_dataset()

get_audio_and_label("formatted_data/formatted_CitroenC4Picasso/CitroenC4Picasso_35.wav")