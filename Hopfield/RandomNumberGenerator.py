import random
import numpy as np

Generator = np.array(range(0, 63))
random.shuffle(Generator)

print(Generator)

exportData = open("RandomNumbers.txt", "w")
exportData.write("\n".join(map(str, Generator)))
exportData.close()
