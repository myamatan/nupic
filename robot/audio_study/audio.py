import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import sys

import pyaudio
import struct

class PlotWindow:

    def __init__(self):
        self.win=pg.GraphicsWindow()
        self.win.setWindowTitle('Realtime plot')
        self.plt=self.win.addPlot() 
        self.plt.setYRange(-1,1)   
        self.curve=self.plt.plot()

        self.CHUNK=1024          
        self.RATE=16000
        self.audio=pyaudio.PyAudio()
        self.stream=self.audio.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=self.RATE,
                                    input=True,
                                    frames_per_buffer=self.CHUNK)

        self.timer=QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)   

        self.data=np.zeros(self.CHUNK)
        xs=np.zeros(self.CHUNK)

    def update(self):
        self.data=np.append(self.data, self.AudioInput())
        if len(self.data)/1024 > 5:  
            self.data=self.data[1024:]
        self.curve.setData(self.data)

    def AudioInput(self):
        ret=self.stream.read(self.CHUNK)
        ret=np.frombuffer(ret, dtype=np.int16).astype(np.float)
        ret[ret > 0.0] /= float(2**15 - 1)
        ret[ret <= 0.0] /= float(2**15) 
        return ret

if __name__=="__main__":
    plotwin=PlotWindow()
    if (sys.flags.interactive!=1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
