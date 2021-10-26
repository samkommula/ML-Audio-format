import librosa
import librosa.display
import matplotlib.pyplot as plt


data_path = "./data/data1.wav"
data_path2 = "./data/data2.wav"


def run():
    time_series, sample_rate = librosa.load(data_path)      # add option 'duration' to change the audio length
    print('The audio is ' + str(len(time_series)/sample_rate)+' seconds')
    x = 0
    while x < len(time_series):
        name = 'time' + str(int(x/sample_rate)) + '-' + str(int(x/sample_rate+1))
        plt.figure(figsize=(10, 5))
        plt.title('Signal waveform')
        plt.ylabel('signal pressure (dB)')
        dB_series = librosa.amplitude_to_db(time_series[x:x+sample_rate], ref=1.0, amin=1e-05)
        librosa.display.waveplot(dB_series, sr=sample_rate, offset=int(x/sample_rate))
        plt.savefig('./result/' + name + '.png')
        plt.close()
        x = x + sample_rate
        print(name+' finished')


if __name__ == '__main__':
    run()


