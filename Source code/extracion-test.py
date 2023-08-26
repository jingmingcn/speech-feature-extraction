from feature_extraction_utils import  *
import pandas as pd

## Read in Sample Sound File (.wav)
sound_filepath = 'example.wav'
sound = parselmouth.Sound(sound_filepath)
df = pd.DataFrame()

##Extract Features
attributes = {}

intensity_attributes = get_intensity_attributes(sound)[0]
pitch_attributes = get_pitch_attributes(sound)[0]
attributes.update(intensity_attributes)
attributes.update(pitch_attributes)

hnr_attributes = get_harmonics_to_noise_ratio_attributes(sound)[0]
gne_attributes = get_glottal_to_noise_ratio_attributes(sound)[0]
attributes.update(hnr_attributes)
attributes.update(gne_attributes)

df['local_jitter'] = None
df['local_shimmer'] = None
df.at[0, 'local_jitter'] = get_local_jitter(sound)
df.at[0, 'local_shimmer'] = get_local_shimmer(sound)

spectrum_attributes = get_spectrum_attributes(sound)[0]
attributes.update(spectrum_attributes)

formant_attributes = get_formant_attributes(sound)[0]
attributes.update(formant_attributes)

lfcc_matrix, mfcc_matrix = get_lfcc(sound), get_mfcc(sound)
df['lfcc'] = None
df['mfcc'] = None
df.at[0, 'lfcc'] = lfcc_matrix
df.at[0, 'mfcc'] = mfcc_matrix

delta_mfcc_matrix = get_delta(mfcc_matrix)
delta_delta_mfcc_matrix = get_delta(delta_mfcc_matrix)
df['delta_mfcc'] = None
df['delta_delta_mfcc'] = None
df.at[0, 'delta_mfcc'] = delta_mfcc_matrix
df.at[0, 'delta_delta_mfcc'] = delta_delta_mfcc_matrix

for attribute in attributes:
    df.at[0, attribute] = attributes[attribute]

df.at[0, 'sound_filepath'] = sound_filepath
rearranged_columns = df.columns.tolist()[-1:] + df.columns.tolist()[:-1]
df = df[rearranged_columns]

##Visualize Features

print(df)
df.to_csv("examp.csv")