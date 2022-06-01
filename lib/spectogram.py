import librosa
import matplotlib.pyplot as plt
import librosa.display


class CreateSpectogram(object):
    def __init__(self, audioPath, imagePath):
        self.x, self.sr = librosa.load(audioPath)
        self.audioPath = audioPath
        self.imagePath = imagePath

    def create(self):
        librosa.load(self.audioPath, sr=None)
        X = librosa.stft(self.x)
        Xdb = librosa.amplitude_to_db(abs(X))
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        librosa.display.specshow(Xdb, sr=self.sr)
        ax.axis('off')
        fig.savefig(self.imagePath, dpi=300, frameon='false')
        plt.close()
