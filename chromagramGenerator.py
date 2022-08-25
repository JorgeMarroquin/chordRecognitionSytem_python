
import jams 
import matplotlib
import numpy as np
import madmom
import os
from random import randint
matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

allSongs = next(os.walk("./songs"), (None, None, []))[2]
allowChords = ["N",
    "A_maj", "A_min", "A#_maj", "A#_min",
    "B_maj", "B_min",
    "C_maj", "C_min", "C#_maj", "C#_min",
    "D_maj", "D_min", "D#_maj", "D#_min",
    "E_maj", "E_min",
    "F_maj", "F_min", "F#_maj", "F#_min",
    "G_maj", "G_min", "G#_maj", "G#_min",
    ]

#Create folders
if(not os.path.exists("./chromagrams")):
    os.mkdir("./chromagrams")
if(not os.path.exists("./chromagrams/test")):
    os.mkdir("./chromagrams/test")
if(not os.path.exists("./chromagrams/train")):
    os.mkdir("./chromagrams/train")

for songName in allSongs:
    print("Generating chromagram for " + songName)
    jam = jams.load("./jams/" + songName[:-4] + ".jams")
    chordAnnotations = jam['annotations'][0]['data']

    dcp = madmom.audio.chroma.DeepChromaProcessor()
    chroma = dcp('./songs/' + songName)

    for index, a in enumerate(chordAnnotations):
        folder = str(a[2]).replace(":", "_")
        trainOrTest = "test/" if randint(1, 4) == 1 else "train/"
        '''if("/" in folder):
            folder = folder[:-folder.index("/")]'''
        if(folder in allowChords):
            print("Chord " + folder + " save in " + trainOrTest)
            savePath = "./chromagrams/" + trainOrTest + folder
            if(not os.path.exists(savePath)):
                os.mkdir(savePath)

            #Plot and save chromagram pieces
            fig, ax = plt.subplots()
            ax.imshow(np.swapaxes(chroma,0,1), aspect='auto', cmap='inferno')
            plt.xlim([(a[0])*10, ((a[0]) + (a[1]))*10])
            plt.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(savePath + "/" + songName[:-4] + "_" + str(int(a[0])) + ".png", bbox_inches = 'tight',
                pad_inches = 0)
            #plt.show()
            plt.close('all')
            plt.clf()
            plt.cla()