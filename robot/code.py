import sys, time
import thread

import nupic
import numpy as np

import soundfile as sf
import pyaudio as pa
import cv2

from collections import deque
import time

import psutil
import matplotlib.pyplot as plt

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.model_factory import ModelFactory

import model_params

SECONDS_PER_STEP = 1/100. #1/16000
WINDOW = 1000


# global
xs = np.array([0]*10240)

def callback(in_data, frame_count, time_info, status):
    global xs
    in_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float)
    in_float[in_float > 0.0] /= float(2**15 - 1)
    in_float[in_float <= 0.0] /= float(2**15) 
    xs = np.r_[xs, in_float]
    xs = xs[-10240:]
    return (in_data, pa.paContinue)

def FFT_AMP(data):
    data=np.hamming(len(data))*data
    data=np.fft.fft(data)
    data=np.abs(data)
    return data

if __name__ == "__main__":

    # pyaudio
    p_in = pa.PyAudio()
    py_format = p_in.get_format_from_width(2)
    fs = 16000 #16000 / 44100
    channels = 1
    chunk = 1024
    use_device_index = 0

    delta = 0

    # turn matplotlib interactive mode on (ion)
    plt.ion()
    fig = plt.figure()
    # plot title, legend, etc
    plt.title('CPU prediction example')
    plt.xlabel('time [s]')

    """Poll CPU usage, make predictions, and plot the results. Runs forever."""
    # Create the model for predicting CPU usage.
    model = ModelFactory.create(model_params.MODEL_PARAMS)
    model.enableInference({'predictedField': 'cpu'})
    # The shifter will align prediction and actual values.
    shifter = InferenceShifter()
    # Keep the last WINDOW predicted and actual values for plotting.
    actHistory = deque([0.0] * WINDOW, maxlen=WINDOW)
    predHistory = deque([0.0] * WINDOW, maxlen=WINDOW)

    # Initialize the plot lines that we will update with each new record.
    actline, = plt.plot(range(WINDOW), actHistory)
    predline, = plt.plot(range(WINDOW), predHistory)
    # Set the y-axis range.
    actline.axes.set_ylim(-1, 30)
    predline.axes.set_ylim(-1, 30)

    in_stream = p_in.open(format=py_format,
                          channels=channels,
                          rate=fs,
                          input=True,
                          output=False,
                          frames_per_buffer=chunk,
                          input_device_index=use_device_index,
                          stream_callback=callback)

    in_stream.start_stream()

    previous_fft=np.array([])

    # input loop
    while in_stream.is_active():
        
        s = time.time()
            
        # Get the CPU usage.
        fft_data=FFT_AMP(xs)
        if fft_data.shape > 10 : previous_fft = fft_data
        else : fft_data = previous_fft
        fft_axis=np.fft.fftfreq(len(xs), d=1.0/2200)
        #print 'fft_data.shape:', fft_data.shape, 'fft_axis.shape:', fft_axis.shape, 'fft_data_amax:', np.argmax(fft_data)
        cpu = fft_data[0:2048].mean() * 1e+2
        ff1 = fft_data[0:2048].mean() * 1e+2
        ff2 = fft_data[2048:4096].mean() * 1e+2
        ff3 = fft_data[4096:6144].mean() * 1e+2
        ff4 = fft_data[6144:8192].mean() * 1e+2
        ff5 = fft_data[8192:10240].mean() * 1e+2
        print( round(ff1,3), round(ff2,3), round(ff3,3), round(ff4,3), round(ff5,4) )

        # Run the input through the model and shift the resulting prediction.
        modelInput = {'cpu': cpu}
        result = shifter.shift(model.run(modelInput))

        # Update the trailing predicted and actual value deques.
        inference = result.inferences['multiStepBestPredictions'][5]
        if inference is not None:
          actHistory.append(result.rawInput['cpu'])
          predHistory.append(inference)

        # Redraw the chart with the new data.
        actline.set_ydata(actHistory)  # update the data
        predline.set_ydata(predHistory)  # update the data
        plt.draw()
        plt.legend( ('actual','predicted') )


        try:
          plt.pause(SECONDS_PER_STEP)
        except:
          pass

    #sf.write("./pyaudio_output.wav", xs, fs)
    in_stream.stop_stream()
    in_stream.close()
    p_in.terminate()

    thread.start_new_thread(loop, ())

