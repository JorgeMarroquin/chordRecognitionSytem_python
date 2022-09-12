import matplotlib.pyplot as plt
import numpy as np
import os

folder = "../chromagramDataset"
labels = next(os.walk(folder + "/train"), (None, None, []))[1]
train = []
test = []

for c in labels:
    train.append(len(next(os.walk(folder + "/train/" + c), (None, None, []))[2]))
    test.append(len(next(os.walk(folder + "/test/" + c), (None, None, []))[2]))

train = np.array(train)
test = np.array(test)

print(np.sum(train), np.sum(test), np.sum(train) + np.sum(test))
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train, width, label='Entrenamiento')
rects2 = ax.bar(x + width/2, test, width, label='Validación')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Cantidad')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=1)
ax.bar_label(rects2, padding=1)

fig.tight_layout()
fig.set_figwidth(13)
plt.xticks(rotation=90)
plt.savefig("../assets/finalDataset.png", bbox_inches = 'tight',
            pad_inches = 0.2)
plt.show()