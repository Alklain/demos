

# instantiate the pipeline
from pyannote.audio import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
from pyannote.database.util import load_rttm

hf_token = 'hf_***'

import torch
print(torch.cuda.is_available())
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.0",
  use_auth_token=hf_token )

a_path = 'data/demo_diar_4.wav'
diarization = pipeline(a_path)

# dump the diarization output to disk using RTTM format
with open("audio_dem.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

groundtruth = load_rttm('C:/wrk/sound_diarisation/audio_dem.rttm')

import numpy as np
df1 = pd.read_csv('C:/wrk/sound_diarisation/audio_dem.rttm', sep=' ', names=['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])
df1.plot(x='p4', y='p8', kind='scatter', rot='vertical')
plt.show()
