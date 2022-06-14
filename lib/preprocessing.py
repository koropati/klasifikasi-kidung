import librosa


class CleanAudio(object):
    def __init__(self, audioInput, audioOutput, dbTreshold, duration, shiftDistance):
        self.audio, self.sr = librosa.load(audioInput, sr=8000, mono=True)
        self.audioInput = audioInput
        self.audioOutput = audioOutput
        self.dbTreshold = dbTreshold
        self.duration = duration
        self.shiftDistance = shiftDistance

    def clean(self):
        # remove all silent with treshold value
        clips = librosa.effects.split(self.audio, top_db=self.dbTreshold)
        # combine all audio without silent data
        wav_data = []
        for c in clips:
            print(c)
            data = self.audio[c[0]: c[1]]
            wav_data.extend(data)
        
        #split to n second duration with shiftDistance duration
        lengthDuration = librosa.get_duration(y=wav_data, sr=self.sr)
        
        loopingLength = int(((self.duration * ((lengthDuration/self.duration)-1)) + self.shiftDistance)/self.shiftDistance)
        
