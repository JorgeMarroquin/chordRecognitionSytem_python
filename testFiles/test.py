import os
allJamFiles = next(os.walk("./jams"), (None, None, []))[2]
count = 0
for jam in allJamFiles:
    print(count, jam[:-5])
    count += 1