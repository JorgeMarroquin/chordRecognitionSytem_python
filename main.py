from distutils.cmd import Command
from email.headerregistry import BaseHeader
from logging import root
import matplotlib
import numpy as np
import madmom
import tensorflow as tf
import os
import shutil
import subprocess
import pytube
import json
import jams
from mutagen.wave import WAVE
from tkinter import * 
from tkinter.filedialog import asksaveasfile
from tkinter.filedialog import askopenfilename
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
        print(yt.length)
        vids= yt.streams.all()
        downloadDir = "./tempSong"
        default_filename = vids[1].default_filename.replace(" ", "_").replace("(", "").replace(")", "")
        default_filename = default_filename.replace("&", "and")
        vids[1].download(downloadDir, default_filename)
        print(default_filename)
        #convert to wav
        command = "ffmpeg -i " + downloadDir + "/" + default_filename + " -ac 2 -f wav " + downloadDir + "/songTemp.wav"
        subprocess.call(command, shell=True)
        os.remove(downloadDir + "/" + default_filename)
        return yt.length, vids[1].default_filename[:-4]
    except Exception as e:
        print("Can't download or convert video")

def convertLocalFile():
    if(not os.path.exists("./tempSong")):
        os.mkdir("./tempSong")
    root = Tk()
    root.geometry('200x150')
    file = askopenfilename(filetypes= (("mp3","*.mp3"), ("mp4","*.mp4"), ("wav","*.wav")))
    root.destroy()
    command = 'ffmpeg -i "' + file + '" ./tempSong/songTemp.wav'
    subprocess.call(command, shell=True)
    audio = WAVE("./tempSong/songTemp.wav")
    return audio.info.length, os.path.basename(file)

def generatePartChromagrams(chroma, beatList, savePath, endSecond):
    if(not os.path.exists(savePath)):
        os.mkdir(savePath)
    beatArray = beatList.tolist()
    beatArray.append(endSecond)
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

    originalChords = []
    chords = []
    scores = []
    new_model = tf.keras.models.load_model('./saved_model')

    loop = tqdm(total=len(beats), position=0, leave=False)
    for index, chroma  in enumerate(beats.copy()):
        
        chroma_url = "./tempChromagrams/" + str(chroma) + ".png"

        img = tf.keras.utils.load_img(chroma_url, target_size=(480, 640))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = new_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        loop.set_description((str(class_names[np.argmax(score)]) + " " + "Score: " + str(100 * np.max(score))).format(index))
        loop.update(1)
        chords.append(class_names[np.argmax(score)].replace("_maj", "").replace("_min", "m"))
        originalChords.append(class_names[np.argmax(score)].replace("_", ":"))
        scores.append(np.max(score))
    loop.close
    return chords, scores, originalChords

def saveFileChords(beats, chords, originalChords, score, title, url, duration):
    root = Tk()
    root.geometry('200x150')
    files = [('JSON', '*.json'), ('jams', '*.jams')]
    file = asksaveasfile(filetypes = files, defaultextension = files)
    root.destroy()
    if(file == None):
        return
    if file.name[-4:] == "json":
        jsonFile = {}
        for index, b in enumerate(beats):
            jsonFile[str(b)] = {"value": str(chords[index]), "score": str(score[index])}
        print(file.name[-4:])
        file.write(json.dumps(jsonFile, indent=4))
    elif file.name[-4:] == "jams":
        jam = jams.JAMS()
        jam.file_metadata.duration = duration
        ann = jams.Annotation(namespace='chord', time=0, duration=jam.file_metadata.duration)
        for index, b in enumerate(beats):
            deltaDuration = beats[index + 1] + 1 if index + 1 < len(beats) else duration
            ann.append(time=b, duration=(deltaDuration - b), confidence=score[index], value=originalChords[index])
        ann.annotation_metadata = jams.AnnotationMetadata(data_source='https://github.com/JorgeMarroquin/chordRecognitionSytem_python')
        jam.file_metadata.identifiers = {"youtube_url": url}
        jam.file_metadata.title = title
        jam.annotations.append(ann)
        jam.save(file.name)

def playSong(beats, chords, title):
    beatIndex = 0
    buffer = ["--", "--", "--", "--"]
    for index, e in enumerate(buffer):
        if index < len(chords):
            buffer[index] = chords[index]

    playback = Playback()
    playBackclick = Playback()
    playback.load_file('./tempSong/songTemp.wav')
    playBackclick.load_file('./assets/click.wav')
    print("Now playing: ", title)
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

    print("===================")
    print("YouTube Url: 1")
    print("Local File:  2")
    fileSrc = input("Select an option: ")
    if fileSrc == "1":
        url = input("Paste youtube link: ")
        songLenght , title = downloadSongFromYoutube(url)
    elif fileSrc == "2":
        songLenght , title = convertLocalFile()
    else:
        print("Exit")
        return

    print("Generating beats")
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(pathTempSong + "/" + songName)
    beat_times = proc(act)
    
    print("Generating chroma")
    dcp = madmom.audio.chroma.DeepChromaProcessor()
    chroma = dcp(pathTempSong + "/" + songName)

    print("Cut chroma")
    generatePartChromagrams(chroma, beat_times, pathImages, songLenght)
    
    print("Get chords from model")
    chords, scores, originalChords = getChords(beat_times)

    loopMenu = True
    while loopMenu:
        print("===================")
        print("Save chords: 1")
        print("Play song:  2")
        print("Exit:       3")
        
        menuOption = input("Select an option: ")
        if(menuOption == "1"):
            saveFileChords(beat_times, chords, originalChords, scores, title, url, songLenght)
        elif(menuOption == "2"):
            playSong(beat_times, chords, title)
        elif(menuOption == "3"):
            print("Exit")
            loopMenu = False
        else:
            print("Option " + menuOption + " not recogniced")

main()