import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("working on gpu")
else:
    device = torch.device("cpu")
    print("working on cpu")




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


model = Classifier()
model.apply(weights_init)
model.to(device)

torch.save(model, './model_file/global_model/server-global-model_0.pth') # global model 저장, 최초 1회만


import socket
from os.path import exists
import select
import sys
import os
# 서버에서 클라이언트로 global model 전송
#클라이언트와의 접속 확인
# 클라이언트로 부터 모델 파일 명 recv
# 클라이언트로부터 모델 파일 recv

#filename = 'server-global-eff-model.pth'

current_path = os.getcwd()
current_model_file_path = os.path.join(current_path, 'model_file')
update_iter = 10

client_model_file_list = []

HOST = 'SERVER IP'
PORT = 'SERVER PORT'
u_index = 0


for k in range(0,update_iter+1):

    server_model_file_name = os.path.join('server-global-model_', str(k), '.pth')
    global_model = Classifier()
    global_model.apply(weights_init)
    global_model.to(device)

    if k == 0:
        torch.save(global_model, './model_file/global_model/server-global-model_' + str(
            k) + '.pth')  ## => 0일떄




    print(f'====================== 클라이언트로부터 global model file {k}번쨰 전송 요청 받음  ============================')


    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # s는 서버 소켓임!
        s.bind((HOST, PORT)) # 서버 자신의 주소와, 클라이언트와의 통로인 포트번호를 설정한다.
        s.listen() # 서버 소켓 s는 클라이언트와 연결을 시작할 수 있도록 바인딩된 포트를 연다.
        print(f'서버가 {k}번째 시작 됩니다.')

        readsocks = [s]
        c = 0


        while True:

            #if len(client_model_file_list) == 2:

            #    break

            readables, writeables, excpetions = select.select(readsocks, [],
                                                              [])  # readsocks에 포함된 소켓에서 이벤트가 발생하는지 감시하는 역할을 한다.



            for sock in readables: # readables는 수신한 데이터를 가진 소켓을 의미한다.

                if sock == s: # 신규 클라이언트 접속 , 만약 서버소켓으로부터 클라이언트의 요청이 들어온다면 readables에는 s가 저장되어있을 것이다.
                    newsock, addr = s.accept() # 클라이언트와 , 서버 사이에 데이터를 주고 받기 위한 통신 소켓 생성, 클라이언트가 접속하면 연결을 수락후 클라이언트의 소켓을 newsock에 저장, 클라이언트의 주소를 addr에 저장
                    c += 1 # 몇 번째 클라이언트인지 확인!
                    print(f'클라이언트 {c}가 접속했습니다. addr : {addr}, sockfd : {newsock}')
                    readsocks.append(newsock) # 접속한 클라이언트를 readsocks 배열에 넣음
                else: # 이미 접속한 클라이언트의 요청
                    # readables는 수신한 데이터를 가진 소켓을 의미. 만약 수신한 데이터를 가진 소켓이 클라이언트 소켓이라면 sock=cli sock으로 할당, s는 계속해서 서버의 소켓이다.
                    conn = sock # 이미 접속한 클라이언트의 요청 (클라이언트가 파일 전송을 요청하는 경우)

                    what_signal = conn.recv(1024).decode('utf-8') # 클라이언트의 신호를 2가지로 나눔 (모델 요청 신호, 모델 전송 신호)

                    conn.sendall('ack'.encode('utf-8')) # ack 시그널을 클라이언트에게 다시 보내줌

                    if what_signal == 'model_request': # 글로벌 모델 요청 시그널

                        file_name = conn.recv(1024).decode('utf-8') # 클라이언트로부터 파일 이름 전송 받음
                        print(f'클라이언트 {c}가 파일 {file_name}을 요청함')

                        try:
                            now_dir = os.path.join('./model_file/global_model/', file_name)
                            print(f'클라이언트 {c}로 파일 {file_name} 전송 시작 ')

                            data_transferred = 0
                            with open(now_dir, 'rb') as f:
                                try :
                                    data = f.read(8192)
                                    while data:
                                        data_transferred += conn.send(data)

                                        data = f.read(8192)
                                except Exception as ex:
                                    print(ex)

                                print("전송 완료 %s 전송량 %d" % (file_name, data_transferred))
                                conn.close()# 파일 전송을 완료한 클라이언트의 소켓은 닫아줘야 함.
                                #readables.remove(sock)
                                readsocks.remove(sock) # 클라이언트 접속 해제시 readsocks에서 제거


                                #break # 수정 필요 while 통과하는 break를 만들어야함
                        except FileNotFoundError:
                            print('파일이 없습니다.')
                            sys.exit()
                            break
                    #else:
                    elif what_signal == 'model_transfer': # 클라이언트로부터 모델 전송 시그널을 받으면
                        print("클라이언트가 업데이트한 모델파일을 전송 받는중...")

                        recv_client_model_file_name = conn.recv(1024).decode('utf-8')  # client 이름 전송 받아야함 , 파일 경로 때문에

                        #client_model_file_name = recv_client_name + '_update_model_' + str(k) + '.pt'
                        client_model_file_list.append(recv_client_model_file_name) # client 모델 파일을 리스트에 넣음
                        # client_2_model_file_name = 'client_2_update_model_' + str(k) + '.pt'

                        update_save_dir = os.path.join(current_model_file_path,
                                                       recv_client_model_file_name[
                                                       13:18] + '\\' + recv_client_model_file_name)  # 저장할 경로


                        update_data = conn.recv(8192) # 클라이언트에게 파일을 전송 받음
                        update_data_transferred = 0



                        with open(update_save_dir, 'wb') as f: # 업데이터 모델 파일 내용 저장

                            try:
                                while update_data:
                                    f.write(update_data)
                                    update_data_transferred += len(update_data)
                                    update_data = conn.recv(8192)

                            except Exception as ex:
                                print(ex)
                        print("파일 %s 받기 완료 : 전송량 %d " % (recv_client_model_file_name, update_data_transferred))

                        client_model_file_list = []
                        conn.close()  # 나중에 클라이언트에서 서버로 파라미터 파일 전송할 때를 대비해서 소켓을 지우면 안되나? 아니면 저때만 다시 소켓을 만드나?
                        readsocks.remove(sock)  # 클라이언트 접속 해제시 readsocks에서 제거

                        #break  # while문 통과

                    else: # 시그널이 잘못되었다면
                        print(f"시그널이 잘못되었습니다. 확인하세요.{what_signal}")
                        break


        client_1_model_file_name, client_2_model_file_name = client_model_file_list[0], client_model_file_list[1]
        print(client_1_model_file_name)
        print(client_2_model_file_name)



        if k != 0:  # fedavg 수행 , k가 1 이상일 때 부터 수행

            # 클라이언트로부터 받은 모델 파일 로드

            # 클라이언트에서 학습한 파라미터 파일 로드
            state_dict_1 = torch.load('./model_file/client_1/' + client_1_model_file_name)
            state_dict_2 = torch.load('./model_file/client_2/' + client_2_model_file_name)

            # 글로벌 모델의 w, b  만들기.
            global_fc1_weight = (state_dict_1['fc1.weight'] + state_dict_2['fc1.weight']) / 2
            global_fc2_weight = (state_dict_1['fc2.weight'] + state_dict_2['fc2.weight']) / 2
            global_fc3_weight = (state_dict_1['fc3.weight'] + state_dict_2['fc3.weight']) / 2

            global_fc1_bias = (state_dict_1['fc1.bias'] + state_dict_2['fc1.bias']) / 2
            global_fc2_bias = (state_dict_1['fc2.bias'] + state_dict_2['fc2.bias']) / 2
            global_fc3_bias = (state_dict_1['fc3.bias'] + state_dict_2['fc3.bias']) / 2

            # 글로벌 모델 객체의 w,b 변수에 각 클라이언트의 w,b를 평균낸 값을 할당.
            global_model.fc1.weight = torch.nn.Parameter(global_fc1_weight)
            global_model.fc1.bias = torch.nn.Parameter(global_fc1_bias)

            global_model.fc2.weight = torch.nn.Parameter(global_fc2_weight)
            global_model.fc2.bias = torch.nn.Parameter(global_fc2_bias)

            global_model.fc3.weight = torch.nn.Parameter(global_fc3_weight)
            global_model.fc3.bias = torch.nn.Parameter(global_fc3_bias)

            # 글로벌 모델 파일 저장
            torch.save(global_model, './model_file/global_model/server-global-model_' + str(
                k) + '.pth')  ## => 2개의 클라이언트의 파일을 fedavg한 후 저장 해야함 .
