import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import sys

import pyaudio
import struct


class PlotWindow:

    def __init__(self):
        self.CHUNK=1024   
        self.RATE=2200
        self.update_seconds=50
        self.audio=pyaudio.PyAudio()
        self.stream=self.audio.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=self.RATE,
                                    input=True,
                                    frames_per_buffer=self.CHUNK)

        self.data=np.zeros(self.CHUNK)
        self.axis=np.fft.fftfreq(len(self.data), d=1.0/self.RATE)

        self.win=pg.GraphicsWindow()
        self.win.setWindowTitle("SpectrumAnalyzer")
        self.plt=self.win.addPlot() 
        self.plt.setYRange(0,100)  

        self.timer=QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_seconds) 

    def update(self):
        self.data=np.append(self.data,self.AudioInput())
        if len(self.data)/1024 > 10:
            self.data=self.data[1024:]
        self.fft_data=self.FFT_AMP(self.data)
        self.axis=np.fft.fftfreq(len(self.data), d=1.0/self.RATE)
        self.plt.plot(x=self.axis, y=self.fft_data, clear=True, pen="y")  #symbol="o", symbolPen="y", symbolBrush="b")

    def AudioInput(self):
        ret=self.stream.read(self.CHUNK)
        ret=np.frombuffer(ret, dtype=np.int16).astype(np.float)
        ret[ret > 0.0] /= float(2**15 - 1)
        ret[ret <= 0.0] /= float(2**15) 
        return ret

    def FFT_AMP(self, data):
        data=np.hamming(len(data))*data
        data=np.fft.fft(data)
        data=np.abs(data)
        return data

if __name__=="__main__":
    plotwin=PlotWindow()
    if (sys.flags.interactive!=1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

