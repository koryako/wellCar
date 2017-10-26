import serial,os

port = os.popen('ls /dev/ttyACM*').read()[:-1]
baud = 115200
ser = serial.Serial(port, baud)

while True:
    line = ser.readline()#.decode('utf-8')
    print line







