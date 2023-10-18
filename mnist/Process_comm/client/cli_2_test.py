import socket

HOST = ''
PORT = 33138
BUFFER_SIZE = 1024

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
    c.connect((HOST, PORT))  # 서버로 connect 요청 보냄
    print('서버와 연결 성공!')




    recv_msg = c.recv(BUFFER_SIZE).decode('utf-8')

    print(f'server로부터 받은 msg : {recv_msg}')

    #c.sendall("hello! server child process".encode('utf-8'))