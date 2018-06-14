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

SECONDS_PER_STEP = 1/16000.
WINDOW = 1000


# global
xs = np.array([0])

def callback(in_data, frame_count, time_info, status):
    global xs
    in_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float)
    in_float[in_float > 0.0] /= float(2**15 - 1)
    in_float[in_float <= 0.0] /= float(2**15) 
    xs = np.r_[xs, in_float]

    return (in_data, pa.paContinue)

if __name__ == "__main__":

    # pyaudio
    p_in = pa.PyAudio()
    py_format = p_in.get_format_from_width(2)
    fs = 16000
    channels = 1
    chunk = 1024
    use_device_index = 0

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
    actline.axes.set_ylim(-200, 200)
    predline.axes.set_ylim(-200, 200)

    in_stream = p_in.open(format=py_format,
                          channels=channels,
                          rate=fs,
                          input=True,
                          output=False,
                          frames_per_buffer=chunk,
                          input_device_index=use_device_index,
                          stream_callback=callback)

    in_stream.start_stream()

    # input loop
    while in_stream.is_active():
        s = time.time()
            
        # Get the CPU usage.
        cpu = round(xs[-1] * 1e+4, 2)
        #cpu = psutil.cpu_percent()

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
        #print 'Actual :', cpu, ', Predicted :', predHistory[-1]
    
    #sf.write("./pyaudio_output.wav", xs, fs)
    in_stream.stop_stream()
    in_stream.close()
    p_in.terminate()

    thread.start_new_thread(loop, ())

