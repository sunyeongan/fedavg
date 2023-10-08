import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from tqdm import tqdm



# 클라이언트와의 접속 확인
# 서버로부터 클로벌 모델 파일 명 recv
# 서버로부터  모델 파일 recv
from socket import *
import socket
from os.path import exists
import sys
import os
from tqdm import tqdm
import time

current_path = os.getcwd()
current_model_file_path = os.path.join(current_path, 'model_file')
update_iter = 32 # 업데이트 횟수

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("working on gpu")
else:
    device = torch.device("cpu")
    print("working on cpu")

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop(self.fc1(out))
        out = self.fc2(out)
        out = self.fc3(out)

        return out


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


HOST = ''
PORT = 33138
client_name = 'client_1'
for k in range(0,update_iter): # ===================모델 업데이트를 10번함 =====================================

    server_model_file_name = 'server-global-model_'+str(k) + '.pth'


    client_model_file_name = 'model-update-cli_1_model_'+str(k)+'.pt'

    print(f'===서버에게 글로벌 모델 파일 혹은, fedavg된 모델 파일을 {k}번째 전송 요청함 =====')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c1_recv_g:
        c1_recv_g.connect((HOST, PORT)) # 서버로 connect 요청 보냄
        print('서버와 연결 성공!')
        while True:




            c1_recv_g.sendall('model_request'.encode('utf-8')) # 모델 요청 신호 전송

            ack_signal = c1_recv_g.recv(1024).decode('utf-8')

            if ack_signal == 'ack':
                print(ack_signal)

                c1_recv_g.sendall(server_model_file_name.encode('utf-8')) # 서버로 글로벌 파일 이름 전송

                data = c1_recv_g.recv(8192) # 서버에게 파일 내용 전송 받음

                data_transferred = 0

                save_dir = os.path.join(current_model_file_path, 'global_model')
                save_dir = os.path.join(save_dir,server_model_file_name) # 글로벌 모델 파일 저장 경로


                with open(save_dir, 'wb') as f: # 글로벌 모델 파일 내용 저장
                    try:
                        while data:
                            f.write(data)
                            data_transferred += len(data)
                            data = c1_recv_g.recv(8192)

                    except Exception as ex:
                        print(ex)
                print("파일 %s 받기 완료 : 전송량 %d " % (server_model_file_name, data_transferred))

                c1_recv_g.close() # 나중에 클라이언트에서 서버로 파라미터 파일 전송할 때를 대비해서 소켓을 지우면 안되나? 아니면 저때만 다시 소켓을 만드나?
                break # while문 통과


    batch_size = 64
    learning_rate = 0.01



    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Download and load the training data
    train_data = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # Download and load the test data
    test_data = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)


    import random

    valid_size = 0.2
    test_size = 0.1
    num_train = len(train_data) # 50000개 중에서 25000개 랜덤 추출

    indices = list(range(num_train)) # 0 ~ 24999
    sample_indices = random.sample(indices, num_train//2) # 50000개 중에서 25000개 랜덤 추출
    len(sample_indices)

    split = int(np.floor(valid_size * (num_train//2))) # 0.2 * 10000
    print(split)
    train_idx, valid_idx = sample_indices[split:], sample_indices[:split] # train 20000, valid 5000 개씩 indices 리스트에서 랜덤한 인덱스 가져오기
    print(len(train_idx), len(valid_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
        sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
        sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)


    #'./model_file_2_cli1/global-update-1-cli1.pt')
    print("======글로벌 모델 파일 로드 ========")
    update_model_path= os.path.join(current_model_file_path,'global_model/'+server_model_file_name)#======
    model = torch.load(update_model_path)
    model.to(device)

    update_cli_model_file_path = os.path.join(current_model_file_path, client_name+'/'+client_model_file_name)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)



    #number of epochs to train the model
    n_epochs = 2

    valid_loss_min = np.Inf  # track change in validation loss

    # keep track of training and validation loss
    train_loss = torch.zeros(n_epochs)
    valid_loss = torch.zeros(n_epochs)

    train_acc = torch.zeros(n_epochs)
    valid_acc = torch.zeros(n_epochs)

    for e in range(0, n_epochs):

        ###################
        # train the model #
        ###################
        model.train()

        torch.cuda.empty_cache()
        for data, labels in tqdm(train_loader):
            # move tensors to GPU if CUDA is available
            data, labels = data.to(device), labels.to(device)

            # clear the gradients of all optimized variables

            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)

            # calculate the batch loss
            loss = criterion(logits, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss[e] += loss.item()

            ps = F.softmax(logits, dim=1)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.reshape(top_class.shape)  # 모델의 출력과, 실제 정답 비교
            train_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()
        train_loss[e] /= len(train_loader)
        train_acc[e] /= len(train_loader)
        scheduler.step()

        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            model.eval()

            torch.cuda.empty_cache()
            for data, labels in tqdm(valid_loader):
                # move tensors to GPU if CUDA is available
                data, labels = data.to(device), labels.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                logits = model(data)
                # calculate the batch loss
                loss = criterion(logits, labels)
                # update average validation loss
                valid_loss[e] += loss.item()

                ps = F.softmax(logits, dim=1)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.reshape(top_class.shape)
                valid_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()

        # calculate average losses
        valid_loss[e] /= len(valid_loader)
        valid_acc[e] /= len(valid_loader)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            e, train_loss[e], valid_loss[e]))

        # print training/validation statistics
        print('Epoch: {} \tTraining accuracy: {:.6f} \tValidation accuracy: {:.6f}'.format(
            e, train_acc[e], valid_acc[e]))

        # save model if validation loss has decreased
        if valid_loss[e] <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss[e]))
            torch.save(model.state_dict(), update_cli_model_file_path)
            valid_loss_min = valid_loss[e]
    model.load_state_dict(torch.load(update_cli_model_file_path))


    # track test loss
    test_loss = 0.0
    test_acc = 0.0

    class_correct = torch.zeros(100)
    class_total = torch.zeros(100)

    model.eval()
    # iterate over test data
    for data, labels in test_loader:
        # move tensors to GPU if CUDA is available
        data, labels = data.to(device), labels.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        logits = model(data)
        # calculate the batch loss
        loss = criterion(logits, labels)
        # update test loss
        test_loss += loss.item()

        ps = F.softmax(logits, dim=1)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.reshape(top_class.shape)
        test_acc += torch.mean(equals.type(torch.float)).detach().cpu()

        for i in range(len(data)):
            label = labels[i]
            class_correct[label] += equals[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss / len(test_loader)
    test_acc = test_acc / len(test_loader)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('Test Accuracy: {:.6f}\n'.format(test_acc))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                torch.sum(class_correct[i]), torch.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    #
    #
    # ## ============================서버로 모델 파라미터 파일 전송 ======================================================
    # # 서버랑 연결 접속 확인
    # # client에서 server로 모델 파일 send
    #
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c1_send_u:
        c1_send_u.connect((HOST, PORT))  # 서버로 connect 요청 보냄
        print('서버와 연결 성공!')
        print('============================서버로 모델 파라미터 파일 전송 ======================================')
        while True:
            c1_send_u.sendall('model_transfer'.encode('utf-8'))  # 모델 전송  신호 전송

            ack_signal2 = c1_send_u.recv(1024).decode('utf-8')

            if ack_signal2 == 'ack':
                print(ack_signal2)

                c1_send_u.sendall(client_model_file_name.encode('utf-8')) # client 1이라는 것을 서버에게 알려주기

                print('파일 %s 전송시작 ! ' % client_model_file_name)
                data_transferred = 0

                with open (update_cli_model_file_path, 'rb') as f:
                    try:
                        data = f.read(8192) # 파라미터 파일에서 8192바이트 만큼 읽어오고
                        while data:
                            data_transferred += c1_send_u.send(data) # data 전송
                            data = f.read(8192) # 다시 읽어옴
                    except Exception as ex:
                        print(ex)
                print("전송완료 : %s 전송량 %d " %(client_model_file_name, data_transferred))

                c1_send_u.close()
                break

# feat bracn 추가 
