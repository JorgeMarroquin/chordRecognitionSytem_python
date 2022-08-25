import numpy as np
import matplotlib.pyplot as plt
import madmom
import librosa
import librosa.display

'''y, sr = librosa.load("./Songs/25.wav", duration=251)
librosa.feature.chroma_stft(y=y, sr=sr)

S = np.abs(librosa.stft(y, n_fft=2048))**2
chroma = librosa.feature.chroma_stft(S=S, sr=sr)'''

dcp = madmom.audio.chroma.DeepChromaProcessor()
chroma = dcp('./Songs/25.wav')

fig, ax = plt.subplots()
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
test = plt.imshow(chroma)
fig.colorbar(test, ax=ax)#.remove()

'''plt.axis('off')
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig("filename.png", bbox_inches = 'tight',
    pad_inches = 0)'''
#plt.xticks(range(0, 252))
#plt.xlim([12.2136965,  12.956735])
plt.show()