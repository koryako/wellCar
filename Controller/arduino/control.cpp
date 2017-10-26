nclude <boarddefs.h>
#include <IRremote.h>
#include <IRremoteInt.h>
#include <ir_Lego_PF_BitStreamEncoder.h>  
int input1 = 5; // 定义uno的pin 5 向 input1 输出   
int input2 = 6; // 定义uno的pin 6 向 input2 输出  
int input3 = 9; // 定义uno的pin 9 向 input3 输出  
int input4 = 10; // 定义uno的pin 10 向 input4 输出  
  
int RECV_PIN=2;
//红外遥控逻辑代码 
#define ADVAN 0x7F6E9C4D  //遥控器>>键
#define BAC   0xFADA21C9      //遥控器<<键
#define STO   0x6165704E  //遥控器>|键
IRrecv irrecv(RECV_PIN);

decode_results results;
  
void setup() {  
 Serial.begin (9600);  
//初始化各IO,模式为OUTPUT 输出模式 
 irrecv.enableIRIn(); 
pinMode(input1,OUTPUT);  
pinMode(input2,OUTPUT);  
pinMode(input3,OUTPUT);  
pinMode(input4,OUTPUT);  
  
}  
  
void loop() {
   if (Serial.available() > 0) {
    delay(100); // 等待数据传完
    int numdata = Serial.available();
    Serial.print("Serial.available = :");
    Serial.println(numdata);
  }
  while(Serial.read()>=0){} //清空串口缓存

 if (irrecv.decode(&results))
   {
    Serial.print("irCode: ");            
   Serial.print(results.value, HEX); // 显示红外编码
   Serial.print(",  bits: ");           
    Serial.println(results.bits); // 显示红外编码位数
     switch(results.value)
     {
     case ADVAN://>>键前进
      //Serial.write(45); // send a byte with thevalue 45
     Serial.write("go"); //sendthe string “hello” and return the length of the string.

       //goForward();
       break;
     case BAC://<<键后退
      Serial.write("back");
     //  back();
       break;
     case STO://>|键停止
      Serial.write("stop");
      // stop();
       break;         
    default:
       delay(600);
     }
     irrecv.resume(); // 接收下一个值
   }
   delay(600);
}

  void stop()
  {
    //stop 停止  
 digitalWrite(input1,LOW);  
 digitalWrite(input2,LOW);    
 digitalWrite(input3,LOW);  
 digitalWrite(input4,LOW);    
 delay(500);  //延时0.5秒  
   Serial.println("stop");
    }
  void goForward()
  {
    digitalWrite(input1,HIGH); //给高电平  
  digitalWrite(input2,LOW);  //给低电平  
  digitalWrite(input3,HIGH); //给高电平  
  digitalWrite(input4,LOW);  //给低电平  
  delay(600);   //延时1秒  
   Serial.println("go");      
    }

    void back()
    {

       //back 向后转  
  digitalWrite(input1,LOW);  
  digitalWrite(input2,HIGH);    
  digitalWrite(input3,LOW);  
  digitalWrite(input4,HIGH);    
  delay(600);      
   Serial.println("back");        
      }

