
import jams 
import matplotlib.pyplot as plt
import numpy as np
import madmom

jam = jams.load("./jams/25.jams")
chordAnnotations = jam['annotations'][0]['data']

#C=Do, D=Re, E=Mi, F=Fa, G=Sol, A=La, B=Si
# '#'=sustained
# 'b'=Bemol
musicalNotes = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]

dcp = madmom.audio.chroma.DeepChromaProcessor()
chroma = dcp('./songs/25.wav')
fig, ax = plt.subplots()
ax.imshow(np.swapaxes(chroma,0,1), aspect='auto', cmap='inferno')
ax.set_yticks(range(0,12))
ax.set_yticklabels(musicalNotes)
#plt.xlim([13.676554000000001*10, 14.442812*10])
plt.savefig("./images/25PartChroma.png", bbox_inches = 'tight', pad_inches = 0.1)
plt.show()