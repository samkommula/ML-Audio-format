# sound_classification

data were stored in directory `data`

## 1, `data_process.py`
this file can read the audio file in `data` and output images of "signal pressure (dB) vs time". Each image represents the signal pressure value in 1 second interval. The output results are stored in `result`

## 2, `sound_classification.py`
extract the MCFF and chroma features from the input audio and do the sound classification. This file is a simple demo based on the provided 2 audios. Dataset could be enlarged, and CNN could be further designed to get better performance.
