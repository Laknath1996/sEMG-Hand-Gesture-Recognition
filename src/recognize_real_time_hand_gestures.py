"""
objective : Execute the real-time hand gesture recognition
author(s) : Ashwin de Silva, Malsha Perera
date      : 
"""

from tma.models.real_time_prediction import *

model_path = 'models/subject_1001_Ashwin/model/cnn_model.h5'
myo.init('/Users/ashwin/FYP/sdk/myo.framework/myo')  # enter the path of the sdk/myo.famework/myo
hub = myo.Hub()
el = EmgLearn(fs=200,
              no_channels=8,
              obs_dur=0.400)
listener = EmgCollector(n=512)

gesture_dict = {
    'Middle_Flexion': 0,
    'Ring_Flexion': 1,
    'Hand_Closure': 2,
    'V_Flexion': 3,
    'Pointer': 4,
    'Neutral': 5,
    'No_Gesture': 6
}

live = Predict(listener=listener,
               emgLearn=el,
               gesture_dict=gesture_dict,
               cnn_model_path=model_path)

with hub.run_in_background(listener.on_event):
    live.main()

print("Closing...")
