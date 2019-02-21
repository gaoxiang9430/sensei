import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.backends.backend_pdf import PdfPages

font = {'family' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

class CIFAR1:
    def __init__(self):
        self.x=[50,120,200]
    def replace30(self):
        return [0.102, 0.251, 0.522]

    def w10(self):
        return [0.118, 0.360, 0.686]

    def ga(self):
        return [0.126, 0.362, 0.732]

class CIFAR2:
    def __init__(self):
        self.x=[50,120,200]
    def replace30(self):
        return [0.102, 0.189, 0.352]

    def w10(self):
        return [0.103, 0.254, 0.486]

    def ga(self):
        return [0.100, 0.245, 0.560]


class GTSRB1:
    def __init__(self):
        self.x=[10,20,30]
        
    def replace30(self):
        return [0.21654548786, 0.56465213254, 0.75890736342]

    def w10(self):
        return [0.21544564654, 0.58541521321, 0.83705463129]

    def ga(self):
        return [0.21544564654, 0.63564512313, 0.88527868033]

class GTSRB2:
    def __init__(self):
        self.x=[10,20,30]
        
    def replace30(self):
        return [0.1398, 0.2221, 0.4295]

    def w10(self):
        return [0.1231, 0.2225, 0.5855]

    def ga(self):
        return [0.1176, 0.2315, 0.673]


fig = plt.figure(figsize=(10, 7))
dataset = CIFAR1()
plt.xlabel('epoch', fontsize=20)
plt.ylabel('training accuracy', fontsize=20)
plt.figure(1)
ax=plt.subplot(111)
#ax.set_aspect(, 0.6)
#plt.axis([0, 1, -, 0.05, 1])
line1, = plt.plot(dataset.x, dataset.replace30(), 'g', label="replace30", linewidth=3.0)
line2, = plt.plot(dataset.x, dataset.w10(), 'r--', label="worst-of-10", linewidth=3.0)
line3, = plt.plot(dataset.x, dataset.ga(), 'b:', label="GA(loss)", linewidth=3.0)
plt.legend()
plt.show()

