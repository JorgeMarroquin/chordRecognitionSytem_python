import matplotlib
import numpy as np
import madmom
import tensorflow as tf
import os
import shutil
import IPython.display as ipd
import subprocess
import pytube
from moviepy.editor import *
matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

'''proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()('./songs/train1.wav')

beat_times = proc(act)'''

def downloadSongFromYoutube(url):
    if(not os.path.exists("./tempSong")):
        os.mkdir("./tempSong")
    try:
        #get youtube video
        yt = pytube.YouTube(url)
        vids= yt.streams.all()
        downloadDir = "./tempSong"
        default_filename = vids[2].default_filename.replace(" ", "_")
        default_filename = default_filename.replace("&", "and")
        vids[2].download(downloadDir, default_filename)
        print(default_filename)
        #convert to wav
        command = "ffmpeg -i " + downloadDir + "/" + default_filename + " -ab 160k -ac 2 -ar 44100 -vn " + downloadDir + "/songTemp.wav"
        subprocess.call(command, shell=True)
        os.remove(downloadDir + "/" + default_filename)
    except Exception as e:
        print("Can't download or convert video")

def generatePartChromagrams(chroma, beatArray, savePath):
    if(not os.path.exists(savePath)):
        os.mkdir(savePath)

    for index, a in enumerate(beatArray):
        print(index, len(beatArray))
        if(index == len(beatArray) - 1):
            return
        #Plot and save chromagram pieces
        fig, ax = plt.subplots()
        ax.imshow(np.swapaxes(chroma,0,1), aspect='auto', cmap='inferno')
        plt.xlim([a*10, (beatArray[index + 1])*10])
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(savePath + "/" + str(a) + ".png", bbox_inches = 'tight',
            pad_inches = 0)
        #plt.show()
        plt.close('all')
        plt.clf()
        plt.cla()
    

def main():
    '''pathTempSong = './tempSong'
    pathImages = './tempChromagrams'
    songName = "songTemp.wav"

    if(os.path.exists(pathTempSong)):
        shutil.rmtree(pathTempSong)
    if(os.path.exists(pathImages)):
        shutil.rmtree(pathImages)

    print("Downloading and converting link")
    url = input("Paste youtube link: \n")
    downloadSongFromYoutube(url)

    print("Generating beats")
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(pathTempSong + "/" + songName)
    beat_times = proc(act)

    print("Generating chroma")
    dcp = madmom.audio.chroma.DeepChromaProcessor()
    chroma = dcp(pathTempSong + "/" + songName)

    print("Cut chroma")
    generatePartChromagrams(chroma, beat_times, pathImages)'''

    print("model")
    new_model = tf.keras.models.load_model('./saved_model')
    sunflower_url = "./tempChromagrams/8.38.png"
    #sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

    img = tf.keras.utils.load_img(
        sunflower_url, target_size=(480, 640)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print("score", score)

    class_names = ['A#_maj', 'A_maj', 'A_min', 'Ab_min', 'B_maj', 'B_min', 'Bb_min', 'C#_maj', 'C#_min', 'C_maj', 'C_min', 'D#_maj', 'D#_min', 'D_maj', 'D_min', 'E_maj', 'E_min', 'F#_maj', 'F#_min', 'F_maj', 'F_min', 'G#_maj', 'G_maj', 'G_min', 'N']
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

main()