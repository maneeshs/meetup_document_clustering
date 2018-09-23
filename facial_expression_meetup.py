import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# get data set
train_path = 'C:\\Users\jainrah\personal\learning\git-repo\kaggle_dataset\\facial_recognition\\meetup\\fer2013\\fer2013_smaller.csv'
test_path = 'C:\\Users\jainrah\personal\learning\git-repo\kaggle_dataset\\facial_recognition\\meetup\\fer2013\\test.csv'

df = pd.read_csv(train_path, encoding="ISO-8859-1")
pixels = df["pixels"]
emotion = df["emotion"]
total_step = len(pixels)

# import scipy.misc
# scipy.misc.imsave('outfile.jpg', image_array)
def preprocess_image_vec(pixels):
    tensor_list = []
    for img_vec in pixels:
        image = np.array([int(item) for item in img_vec.split(' ')])
        reshaped_image = image.reshape(-1, 48)
        image_tensor = torch.Tensor(reshaped_image).unsqueeze(0)
        tensor_list.append(image_tensor)
    return tensor_list

class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.cn_l1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cn_l2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cn_l3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(in_features=32*4*4, out_features=7)

    def forward(self, x):
        out_cnl1 = self.cn_l1(x)
        out_cnl2 = self.cn_l2(out_cnl1)
        out_cnl3 = self.cn_l3(out_cnl2)
        out_cnl3 = out_cnl3.reshape(out_cnl3.size(0), -1)
        return self.fc1(out_cnl3)

net = CNNet()

#  code for training
if __name__ == '__main__':

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 100
    total_records = len(pixels)
    batch_size = 128

    img_tensors_list = preprocess_image_vec(pixels)

    for epoch in range(num_epochs):
        for i in range(int(total_records/batch_size)):
            image_batch = torch.stack(img_tensors_list[batch_size * i:batch_size * (i + 1)])
            # emotion_batch = torch.stack(emotion_tensors_list[batch_size * i:batch_size * (i + 1)])
            outputs = net(image_batch)
            loss = criterion(outputs, torch.Tensor(list(emotion[batch_size * i:batch_size * (i + 1)])).long())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(i, epoch)
            if (i + 1) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    torch.save(net.state_dict(), 'expression_evaluation.ckpt')

# code for testing
if __name__ == '__main__':
    # Test the model
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    df_test = pd.read_csv(test_path, encoding="ISO-8859-1")
    test_pixels = df_test["pixels"]
    test_emotion = df_test["emotion"]

    img_tensors_list = preprocess_image_vec(test_pixels)
    net.load_state_dict(torch.load('expression_evaluation.ckpt')) #. ##

    with torch.no_grad():
        correct = 0
        total = 0
        for i, images in enumerate(img_tensors_list):
            outputs = net(images.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            # total += test_emotion[].size(0)
            # correct += (predicted == labels).sum().item()
            total += 1
            correct += (predicted.numpy()[0] == test_emotion[i]).sum().item()

        print('Test Accuracy of the model on the {} test images: {} %'.format(len(test_pixels), 100 * correct / total))

# Save the model checkpoint
# torch.save(net.state_dict(), 'net.ckpt')