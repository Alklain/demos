from pyannote.audio import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
from pyannote.database.util import load_rttm
import numpy as np
import numpy as np
import soundfile as sf


from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment
import subprocess
import json
import os
import soundfile as sf
import wave
import sys
import regex
import transformers
import torch
import sklearn
import nltk
SetLogLevel(0)
from scipy.signal import resample
import scipy.io.wavfile as wav
from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузка модели и токенизатора
model_name = "Vikhrmodels/Vikhr-Qwen-2.5-0.5B-Instruct"
lllmmodel = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# исходный акустический сигнал
a_path = 'data/demo_diar_4.wav'
model_path = 'C:/wrk/sound_diarisation/vosk-model-small-ru-0.22'
# Устанавливаем Frame Rate
FRAME_RATE = 16000
CHANNELS = 1

model = Model(model_path)
rec = KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)



pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0" )
diarization = pipeline(a_path)

# dump the diarization output to disk using RTTM format
with open("audio_dem.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

groundtruth = load_rttm('C:/wrk/sound_diarisation/audio_dem.rttm')
df1 = pd.read_csv('C:/wrk/sound_diarisation/audio_dem.rttm', sep=' ', names=['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])
#df1.plot(x='p4', y='p8', kind='scatter', rot='vertical')
#plt.show()

#print(df1[])
print(df1[['p8','p4','p5']])

spk = list(df1['p8'])
ts = list(df1['p4'])
lf = list(df1['p5'])
[data, sr] = sf.read(a_path)
for i in range(len(lf)):
    if lf[i]>2: # вырезаем фрагмент звукового файла
        [d1, sf1] = sf.read(a_path, start=int(sr*float(ts[i])), stop=int(sr*(float(ts[i])+float(lf[i]))))
        sf.write('temp.wav', d1, sf1)
        # Перевести частоту дискретизации
        new_rate = 44100
        # Откройте аудиофайл
        sample_rate, data = wav.read('temp.wav')
        resampled_data = resample(data, int(len(data) * new_rate / sample_rate))
        resampled_data = resampled_data.astype('int16')
        wav.write('audio_resampled.wav', new_rate, resampled_data)


        # распознаем текст
        wf = wave.open('temp.wav', "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            sys.exit(1)

        # Преобразуем вывод текст
        #print('Распознавание в текст')
        rec.AcceptWaveform(wf.readframes(wf.getnframes()))
        result = rec.Result()
        text = json.loads(result)["text"]
        print(spk[i])
        print(text)
        # обрабобка
        messages = [
            {"role": "system", "content": "Вы редактор, исправляете ошибки и опечатки, расставьте знаки препинания в тексте на русском языке"},
            {"role": "user", "content": 'Исправляете ошибки и опечатки, расставьте знаки препинания в тексте на русском языке: '+text+', в ответе дай только исправленную фразу'},
        ]

        # Токенизация и генерация текста
        input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True,
                                                  return_tensors="pt")
        output = lllmmodel.generate(
            input_ids,
            max_length=300,
            temperature=0.2,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=20,
            top_p=0.9,
        )

        # Декодирование и вывод результата
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(generated_text)
    else:
        pass





