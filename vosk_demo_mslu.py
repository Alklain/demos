from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import os
import wave
import sys
SetLogLevel(0)
model_path = 'C:/wrk/sound_diarisation/vosk-model-small-ru-0.22'
#model_path = 'C:/wrk/sound_diarisation/vosk-model-ru-0.42'
# Проверяем наличие модели
if not os.path.exists(model_path):
    print(
        "Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)
# Устанавливаем Frame Rate
FRAME_RATE = 44100
CHANNELS = 1
model = Model(model_path)
rec = KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)
a_path = 'C:/wrk/DSP_2024/One_two.wav'
wf = wave.open(a_path, "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print("Audio file must be WAV format mono PCM.")
    sys.exit(1)
# Преобразуем вывод текст
print('Начали думать')
rec.AcceptWaveform(wf.readframes(wf.getnframes()))
result = rec.Result()
text = json.loads(result)["text"]
print(text)
