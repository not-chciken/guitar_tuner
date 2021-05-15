# Python Guitar Tuner
## Settings
In this repository you can find a simple __python guitar tuner__ based on the _Harmonic Product Spectrum_.
Just execute the `hps_tuner_plt.py` script and enjoy a tuned guitar. \
A thorough explanation of the underlying theoretical concepts can be found on my [website](https://www.chciken.com/digital/signal/processing/2020/05/13/guitar-tuner.html). \
If you want to fine-tune the tuner (hehe), just fiddle with the following script parameters:
``` python
# The sample frequency in Hz. With 48kHz sampling frequency we can capture
# all frequencies from 0Hz to 24kHz which is more than sufficient for most
# instruments.
SAMPLE_FREQ = 48000
# Window size of the DFT in samples.
# A good compromise between frequency resolution and delay is a window
# size of 48000. This corresponds to a 1 second delay and an frequency
# resolution of 1 Hz assuming a sampling frequency of 48kHz
WINDOW_SIZE = 48000
# Number of samples the DFT window step size.
# With 12000 samples every 0.25s a new tuning process is executed.
WINDOW_STEP = 12000
# Maximum number of harmonic product spectrums
NUM_HPS = 8
# Tuning is activated if the signal power exceeds this threshold
POWER_THRESH = 7e-6
# Definition of the concert pitch 'a1'.
CONCERT_PITCH = 440
# Everything under WHITE_NOISE_THRESH*avg_energy_per_freq is cut off.
WHITE_NOISE_THRESH = 0.2
```
## Dependencies
Python dependencies:
```bash 
pip install numpy
pip install scipy
pip install sounddevice
```
On linux you might need:
```
sudo apt-get install libportaudio2
```