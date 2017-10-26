"""
void setup() {  
  Serial.begin(9600);  
  while(Serial.read()>= 0){}//clear serialbuffer  
}  
   
void loop() {  
   if (Serial.available() > 0) {  
    delay(100); // 等待数据传完  
    int numdata = Serial.available();  
    Serial.print("Serial.available = :");  
    Serial.println(numdata);  
  }  
  while(Serial.read()>=0){} //清空串口缓存  
} 


 void setup(){   
Serial.begin(9600);   
}  
void loop(){   
  Serial.write(45); // send a byte with thevalue 45   
  int bytesSent = Serial.write(“hello”); //sendthe string “hello” and return the length of the string.   
}  
"""