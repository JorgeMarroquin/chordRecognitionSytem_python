import jams
import os
import subprocess
import pytube
from moviepy.editor import *

errorCount = 0
errorFiles = []
if(not os.path.exists("./songs")):
    os.mkdir("./songs")
#Get all jam names
allJamFiles = next(os.walk("./jams"), (None, None, []))[2]

for jamName in allJamFiles:
    try:
        #Load jam information
        jam = jams.load("./jams/" + jamName)
        urlYT = jam['file_metadata']['identifiers']['youtube_url']
        #get youtube video
        yt = pytube.YouTube(urlYT)
        vids= yt.streams.all()
        downloadDir = "./songs"
        default_filename = vids[0].default_filename.replace(" ", "_")
        default_filename = default_filename.replace("&", "and")
        vids[0].download(downloadDir, default_filename)
        print(default_filename)
        #convert to wav
        command = "ffmpeg -i " + downloadDir + "/" + default_filename + " -ab 160k -ac 2 -ar 44100 -vn " + downloadDir + "/" + jamName[:-5] + ".wav"
        subprocess.call(command, shell=True)
        os.remove(downloadDir + "/" + default_filename)
    except Exception as e:
        print("##################ERROR##################")
        print(jamName, str(e))
        errorCount += 1
        errorFiles.append(jamName+ str(e) + str("\n"))
with open('songsDownloaderErrorLog.txt', 'w') as f:
    f.write(str(errorCount) + " songs can't be downloaded \n")
    f.write("Error files:\n")
    f.writelines(errorFiles)
