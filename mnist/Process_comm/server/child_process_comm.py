
import socket
from os.path import exists
import select
import sys
import os
from multiprocessing import Process

HOST = ''
PORT = 33138
BUFFER_SIZE = 1024

def child_process(dic, idx):
    # 자식 프로세스에서 부모 프로세스의 변수에 접근하여 값을 변경
    print("Child Process pid : ", os.getpid(), " ppid : ", os.getppid())
    print(f'client dictionary : {dic}')
    cli_sock = list(dic.keys())[idx]
    print(f'{idx} 의 socket :  {list(dic.keys())[idx]}')

    send_msg = 'child' + str(idx) + 'msg'
    cli_sock.sendall(send_msg.encode('utf-8'))
    
    exit(0)

if __name__ == '__main__':

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # s는 서버 소켓임!
            s.bind((HOST, PORT)) # 서버 자신의 주소와, 클라이언트와의 통로인 포트번호를 설정한다.
            s.listen() # 서버 소켓 s는 클라이언트와 연결을 시작할 수 있도록 바인딩된 포트를 연다.
            
            readsocks = [s]
            
            dict = {}
            client_idx = 0

            while True : # 조건 달기 
                readables, writeables, excpetions = select.select(readsocks, [], [])  # readsocks에 포함된 소켓에서 이벤트가 발생하는지 감시하는 역할을 한다.



                for sock in readables: # readables는 수신한 데이터를 가진 소켓을 의미한다.

                    if sock == s: # 신규 클라이언트 접속 , 만약 서버소켓으로부터 클라이언트의 요청이 들어온다면 readables에는 s가 저장되어있을 것이다.
                        newsock, addr = s.accept() # 클라이언트와 , 서버 사이에 데이터를 주고 받기 위한 통신 소켓 생성, 클라이언트가 접속하면 연결을 수락후 클라이언트의 소켓을 newsock에 저장, 클라이언트의 주소를 addr에 저장
                        
                        
                        readsocks.append(newsock) # 접속한 클라이언트를 readsocks 배열에 넣음
                        dict[newsock] = client_idx

                        p = Process(target=child_process, args=(dict,client_idx))
                        p.start()
                        p.join()

                        client_idx += 1



                    # else: # 이미 접속한 클라이언트의 요청
                    #     # readables는 수신한 데이터를 가진 소켓을 의미. 만약 수신한 데이터를 가진 소켓이 클라이언트 소켓이라면 sock=cli sock으로 할당, s는 계속해서 서버의 소켓이다.
                    #     conn = sock # 이미 접속한 클라이언트의 요청 (클라이언트가 파일 전송을 요청하는 경우)