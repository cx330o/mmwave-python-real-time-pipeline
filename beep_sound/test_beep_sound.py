import winsound
import time
duration = 2000  # milliseconds
finish_freq = 440  # Hz
next_experiment = 600
action_freq = 800  # Hz
action_duration = 1000  # milliseconds

def beep(freq, duration, experiment):
        winsound.Beep(freq, duration)
        # Add marker:
        # timeStamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        # beep_q.put([timeStamp, experiment])

beep(finish_freq, duration, '')
time.sleep(2)
beep(next_experiment, duration, '')
time.sleep(2)
beep(action_freq, duration, '')