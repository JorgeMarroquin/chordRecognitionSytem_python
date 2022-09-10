'''from time import sleep
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from playsound import playsound
playsound('../tempSong/songTemp.wav')'''

'''from just_playback import Playback
playback = Playback() # creates an object for managing playback of a single audio file
playback.load_file('../tempSong/songTemp.wav')

playback.play()

while(playback.active):

    print(" " + str(playback.curr_pos), end='\r')
import os
allChromas = next(os.walk("../tempChromagrams"), (None, None, []))[2]
print(allChromas)'''

print('\033[1m' + "hola" + '\033[0m')