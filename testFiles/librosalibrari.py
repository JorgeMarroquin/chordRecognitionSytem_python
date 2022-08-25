import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

y, sr = librosa.load("sample.wav")
c, srch = librosa.load("sample.wav")

D = librosa.stft(y)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

E = librosa.stft(c)  # STFT of y
S_dbe = librosa.amplitude_to_db(np.abs(E), ref=np.max)

plt.figure()
librosa.display.specshow(S_db)
plt.figure()
librosa.display.specshow(S_dbe)
plt.colorbar()
plt.show()