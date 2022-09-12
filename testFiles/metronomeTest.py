from time import sleep
from just_playback import Playback
import threading
playBackclick = Playback()
playBackclick.load_file('../assets/click.wav')

for i in range(0, 60):
    x = threading.Thread(target=playBackclick.play, args=())
    x.start()
    sleep(1)
