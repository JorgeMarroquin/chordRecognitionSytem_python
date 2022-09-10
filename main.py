from time import sleep
import matplotlib
import numpy as np
import madmom
import tensorflow as tf
import os
import shutil
import IPython.display as ipd
import subprocess
import pytube
import json
import time
import keyboard
from just_playback import Playback
from tqdm import tqdm
from moviepy.editor import *
matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

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

    loop = tqdm(total=len(beatArray), position=0, leave=False)
    for index, a in enumerate(beatArray):
        loop.set_description("".format(index))
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
        loop.update(1)
    loop.close

def getChords(beats):
    with open('class_indices.json', 'r') as openfile:
        json_object = json.load(openfile)
    class_names = list(json_object.keys())
    for index, element in enumerate(class_names):
        class_names[index] = element.replace("_maj", "").replace("_min", "m")

    chords = []
    new_model = tf.keras.models.load_model('./saved_model')

    loop = tqdm(total=len(beats), position=0, leave=False)
    for index, chroma  in enumerate(beats[:-1].copy()):
        
        chroma_url = "./tempChromagrams/" + str(chroma) + ".png"

        img = tf.keras.utils.load_img(chroma_url, target_size=(480, 640))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = new_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        loop.set_description((str(class_names[np.argmax(score)]) + " " + "Score: " + str(100 * np.max(score))).format(index))
        loop.update(1)
        chords.append(class_names[np.argmax(score)])
    loop.close
    return chords


def deploySong(beats, chords):
    beatIndex = 0
    buffer = ["--", "--", "--", "--"]
    for index, e in enumerate(buffer):
        if index < len(chords):
            buffer[index] = chords[index]

    playback = Playback()
    playBackclick = Playback()
    playback.load_file('./tempSong/songTemp.wav')
    playBackclick.load_file('./assets/click.wav')
    playback.play()
    while(beatIndex < len(chords)):
        if(round(playback.curr_pos, 2) == beats[beatIndex]):
            print("    " + str(round(playback.curr_pos, 2)) + " - | " + str(buffer[0]) + " | -> " + str(buffer[1]) + " -> " + str(buffer[2]) + " -> " + str(buffer[3] + "         "), end="\r")
            beatIndex += 1
            buffer.pop(0)
            buffer.append(chords[beatIndex+3] if beatIndex + 3 < len(chords) else "--")
            playBackclick.play()

def main():
    
    pathTempSong = './tempSong'
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
    generatePartChromagrams(chroma, beat_times, pathImages)
    
    print("Get chords from model")
    chords = getChords(beat_times)

    deploySong(beat_times, chords)

main()