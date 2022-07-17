import librosa

audio, sr = librosa.load("myaudio.wav", sr=None, mono=True)
audio2, sr2 = librosa.load("myaudio.wav", sr=9000, mono=True)

print("AUDIO 1 LENGTH: {}".format(len(audio)))
print("AUDIO 2 LENGTH: {}".format(len(audio2)))
