#coding=utf-8

import socket  
import struct  
import hashlib,base64  
import threading,random  
#线程,套接字,哈希表,随机数  
  
#存放链接客户fd,元组  
connectionlist = {}  
  
#发送指定的消息  
def sendMessage(message):  
    #对全局变量的引用  
    global connectionlist  
    #向客户端集合中每个成员发送消息  
    # %:字符串合成  
    # ,:字符串叠加  
    for connection in connectionlist.values():  
        connection.send("\x00%s\xFF" % message)  
  
#删除连接,从集合中删除连接对象item(建立一个连接追加到连接中)  
def deleteconnection(item):  
    global connectionlist  
    del connectionlist['connection'+item]  
  
#定义WebSocket对象(基于线程对象)  
class WebSocket(threading.Thread):  
    #  
    def __init__(self,conn,index,name,remote, path="/"):  
        #初始化线程  
        threading.Thread.__init__(self)  
        #初始化数据,全部存储到自己的数据结构中self  
        self.conn = conn  
        self.index = index  
        self.name = name  
        self.remote = remote  
        self.path = path  
        self.buffer = ""  
      
    #运行线程  
    def run(self):  
        #Log输出,套接字index启动  
        print ('Socket%s Start!' % self.index ) 
        headers = {}  
        #Socket是否握手的标志,初始化为false.  
        self.handshaken = False  
        #循环执行如下逻辑  
        while True:  
            #如果没有进行握手  
            if self.handshaken == False:  
                #Log输出,Socket x开始与远程客户y进行握手过程  
                print ('Socket%s Start Handshaken with %s!' % (self.index,self.remote)  )
                #从客户端接受1kb数据，存放到buffer中  
                self.buffer += self.conn.recv(1024)  
                #如果接受数据中有\r\n\r\n的标志  
                if self.buffer.find('\r\n\r\n') != -1:  
                    #按照这种标志分割一次,结果为：header data(网页的解析)  
                    #再对header 和 data部分进行单独的解析  
                    header, data = self.buffer.split('\r\n\r\n', 1)  
                    #对header进行分割后，取出后面的n-1个部分  
                    for line in header.split("\r\n")[1:]:  
                        #逐行的解析Request Header信息(Key,Value)  
                        key, value = line.split(": ", 1)  
                        #然后存放在一个Hash表中,方便访问  
                        headers[key] = value  
                    #人为定义Location的item的信息  
                    #构造location:ws://localhost/path ?  
                    headers["Location"] = "ws://%s%s" %(headers["Host"], self.path)  
                    print ("Location:",headers["Location"] ) 
                    print ("headers:",headers )
                    #取出其中两项信息key1 key2  
                    #key1 = headers["Sec-WebSocket-Key1"]  
                      
                    #key2 = headers["Sec-WebSocket-Key2"]  
                    #Header解析完毕后，分析data部分  
                    #如果data部分长度小于8的话，从客户端那边接续接受数据使得data变为8字节  
                    if len(data) < 8:  
                        data += self.conn.recv(8-len(data))  
                    #将data的数据信息存放为key3变量(前面的第一个八个字节为key3)  
                    #key3 = data[:8]  
                    #将data后面的数据作为buffer进行存放  
                    self.buffer = data[8:]  
                    #根据key1,key2,key3产生token？  
                    #根据客户的key1、key2、8字节的关键字  
                    #产生一个16位的安全秘钥  
                    #old-Protocol  
                    #token = self.generate_token(key1, key2, key3)  
                    #new Protocol  
                    token = generate_token_2(self, key)  
                    #握手过程,服务器构建握手的信息,进行验证和匹配  
                    #Upgrade: WebSocket 表示为一个特殊的http请求,请求目的为从http协议升级到websocket协议  
                    handshake = '\  
                    HTTP/1.1 101 Web Socket Protocol Handshake\r\n\  
                    Upgrade: WebSocket\r\n\  
                    Connection: Upgrade\r\n\  
                    Sec-WebSocket-Origin: %s\r\n\  
                    Sec-WebSocket-Location: %s\r\n\r\n\  
                    ' %(headers['Origin'], headers['Location'])  
                    #服务端发送握手数据 & 根据Key产生的Token值  
                    self.conn.send(handshake+token)  
                    #这个操作之后才设定为握手状态  
                    self.handshaken = True  
                    #Log输出状态：套接字x与客户y握手成功。  
                    print 'Socket%s Handshaken with %s success!' % (self.index,self.remote)  
                    #向全部连接客户端集合发送消息,(环境套接字x的到来)  
                    sendMessage('Welcome, '+self.name+' !')  
            else:  
                #如果已经握手  
                #从客户端读取64字节的数据  
                self.buffer += self.conn.recv(64)  
                #如果数据中存在FF的标志，则按照此标志进行分解  
                if self.buffer.find("\xFF")!=-1:  
                    #分解方式啥含义??  
                    s = self.buffer.split("\xFF")[0][1:]  
                    #如果消息是'终止',则打印Socket退出  
                    if s=='quit':  
                        print 'Socket%s Logout!' % (self.index)  
                        #全体同志Socket退出的状态(进行GUI更新准备)  
                        sendMessage(self.name+' Logout')  
                        #同时删除socket连接集合  
                        deleteconnection(str(self.index))  
                        #同时关闭对应的WebSocket连接(多线程关系)  
                        self.conn.close()  
                        break  
                    else:  
                        #否则输出,Socket x收到客户端的消息 y  
                        print 'Socket%s Got msg:%s from %s!' % (self.index,s,self.remote)  
                        #向全体的客户端输出连接的信息  
                        sendMessage(self.name+':'+s)  
                    #Buffer信息再一次的清空  
                    self.buffer = ""  
  
def generate_token_2(self, key):  
        key = key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'  
        ser_key = hashlib.sha1(key).digest()  
        return base64.b64encode(ser_key)  
              
def generate_token(self, key1, key2, key3):  
    #list/tuple(ls)-list元组相互转换  
    #这句话没看懂,如何理解key是否为数字  
    num1 = int("".join([digit for digit in list(key1) if digit.isdigit()]))  
    #解析key1中空格的个数  
    spaces1 = len([char for char in list(key1) if char == " "])  
    #解析后number2对象  
    num2 = int("".join([digit for digit in list(key2) if digit.isdigit()]))  
    #统计空格的个数?安全性验证  
    spaces2 = len([char for char in list(key2) if char == " "])  
    #按照一定的格式进行打包,然后进行网络传输(格式可以自己进行预订)  
    #struck.pack：http://blog.sina.com.cn/s/blog_4b5039210100f1tu.html  
    combined = struct.pack(">II", num1/spaces1, num2/spaces2) + key3  
    #对打包的值进行md5解码后,并返回二进制的形式  
    ##hexdigest() 为十六进制值，digest()为二进制值  
    #处理MD5: http://wuqinzhong.blog.163.com/blog/static/4522231200942225810117/  
    return hashlib.md5(combined).digest()  
      
#创建WebSocket服务器对象()  
class WebSocketServer(object):  
    #初始化时,socket为空  
    def __init__(self):  
        self.socket = None  
    #开启操作  
    def begin(self):  
        #服务器尽心启动Log输出  
        print 'WebSocketServer Start!'  
        #创建TCP的套接字,监听IP、Port  
        #这里可以自己进行设置,最多可以收听50个请求客户  
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        ip = 'localhost'  
        port = 8080  
          
        #print "WebServer is listening %s,%d" % ip,port  
        #self.socket.bind(ip,port)  
        self.socket.bind((ip,port))  
        self.socket.listen(50)  
        #声明全局连接集合  
        global connectionlist  
          
        i=0  
        while True:  
            #服务器响应请求,返回连接客户的信息(连接fd,客户地址)  
            connection, address = self.socket.accept()  
            #address信息中的第1个字符串为username对象  
            username=address[0]  
            #根据连接的客户信息,创建WebSocket对象(本质为一个线程)  
            #连接后sockfd，连接index，用户名，地址  
            newSocket = WebSocket(connection,i,username,address)  
            #线程启动  
            newSocket.start()  
            #更新连接的集合(hash表的对应关系)-name->sockfd  
            connectionlist['connection'+str(i)]=connection  
            i = i + 1  
  
if __name__ == "__main__":  
    server = WebSocketServer()  
    server.begin()  