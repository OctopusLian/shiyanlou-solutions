#Client.py
import socket              


s = socket.socket()         
host = socket.gethostname() 
port = 12345               
s.connect((host, port))
print "Linked"
info = ""
while info != "exit":
    print "From Others: "+info
    send_mes = raw_input("")
    s.send(send_mes)
    if send_mes == "exit":
        break
    info = s.recv(1024)
s.close()  
