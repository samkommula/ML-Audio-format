import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(1),
        )
        self.out = nn.Linear(8 * 4 * 8, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


def parse_wav(data):
    time_series, sampling_rate = librosa.load(data)
    # default librosa.feature.mfcc return 20 dimension
    mfcc = librosa.feature.mfcc(y=time_series, sr=sampling_rate)
    # default librosa.feature.chroma_stft return 12 dimension
    chroma = librosa.feature.chroma_stft(y=time_series, sr=sampling_rate)
    # scale this data with its mean
    mfcc_feature = np.vstack([scale(np.mean(mfcc, axis=1).T)])
    chroma_feature = np.vstack([scale(np.mean(chroma, axis=1).T)])
    features = np.hstack([mfcc_feature, chroma_feature])
    return features


if __name__ == '__main__':
    data_path = "./data/data1.wav"
    data_path2 = "./data/data2.wav"
    x1 = parse_wav(data_path).reshape(4, 8)
    x2 = parse_wav(data_path2).reshape(4, 8)
    x_ = np.stack((x1, x2), axis=0)
    x_ = np.vstack((x_, x_))
    x_ = np.expand_dims(x_, axis=1)
    y_ = np.array([0, 1, 0, 1])    # enumerate the label
    x_ = torch.from_numpy(x_)
    y_ = torch.from_numpy(y_)
    # duplicate the x_, y_ data only for simple training demo (could be replaced with really data)
    x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.3)
    # cnn
    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    # train
    for i in range(len(x_train)):
        x = torch.from_numpy(np.expand_dims(x_train[i], axis=0))
        y = torch.from_numpy(np.expand_dims(y_train[i], axis=0))
        # (batch, kernel, x_axis, y_axis)
        output = cnn(x)[0]  # cnn output
        loss = loss_func(output, y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()
        optimizer.step()
    # test
    test_output, last_layer = cnn(x_test)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    accuracy = float((pred_y == y_test.data.numpy()).astype(int).sum()) / float(y_train.size(0))
    print('test accuracy: %.2f' % accuracy)