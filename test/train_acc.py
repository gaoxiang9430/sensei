import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.backends.backend_pdf import PdfPages

font = {'family' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

class CIFAR10:
    def __init__(self):
        pass
    def replace30(self):
        f= open("replace30.txt", 'r')
        return f.readlines()

    def w10(self):
        f= open("w_10.txt", 'r')
        return f.readlines()

    def ga(self): #TODO
        f= open("ga.txt", 'r')
        return f.readlines()


fig = plt.figure(figsize=(10, 7))
dataset = CIFAR10()
plt.xlabel('epoch', fontsize=20)
plt.ylabel('training accuracy', fontsize=20)
plt.figure(1)
ax=plt.subplot(111)

#ax.set_aspect(, 0.6)
#plt.axis([0, 1, -, 0.05, 1])
line1, = plt.plot(dataset.replace30(), 'g', label="replace30", linewidth=3.0)
line2, = plt.plot(dataset.w10(), 'r--', label="worst-of-10", linewidth=3.0)
line3, = plt.plot(dataset.ga(), 'b:', label="GA(loss)", linewidth=3.0)
plt.legend()
plt.show()

