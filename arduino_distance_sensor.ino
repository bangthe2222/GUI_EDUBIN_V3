const int trig1 = 3;
const int echo1 = 2;
const int trig2 = 5;
const int echo2 = 4;
String msg;
void readSerialPort();
void setup() 
{
  Serial.begin(9600);
  pinMode(trig1,OUTPUT);
  pinMode(echo1,INPUT);
  pinMode(trig2,OUTPUT);
  pinMode(echo2,INPUT);
  pinMode(6, OUTPUT);
  pinMode(7, OUTPUT);    
  pinMode(8, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);
  pinMode(11, OUTPUT);
  pinMode(12, OUTPUT);  

}

void loop() 
{
  unsigned long duration1;
  int distance1;
  unsigned long duration2;
  int distance2;
  digitalWrite(trig2,0);
  delayMicroseconds(2);
  digitalWrite(trig2,1);
  delayMicroseconds(5);
  digitalWrite(trig2,0);
  duration2 = pulseIn(echo2,HIGH);
  distance2 = int(duration2/2/28);
  if (distance2 <= 50){
    digitalWrite(trig1,0);
    delayMicroseconds(2);
    digitalWrite(trig1,1);
    delayMicroseconds(5);
    digitalWrite(trig1,0);
    duration1 = pulseIn(echo1,HIGH);
    distance1 = int(duration1/2/28);
    if (distance1 < 25){
      Serial.println("d2_1_d1_1");
    }else{
      Serial.println("d2_1_d1_0");
    }
  }else if((distance2 >50) && (distance2 < 100)){
      Serial.println("d2_2");
  }else{
      Serial.println("d2_0");
  }
  Serial.flush();
  readSerialPort();
  if (msg != ""){
    for(int i= 0;i<7;i++){
      int id = i + 6;
      if (msg[i] == '1'){
      digitalWrite(id,HIGH);
      }else{
      digitalWrite(id,LOW);
      }
    }
  }
  Serial.println(msg);
  delay(100);
  

}

void readSerialPort() {
  msg = "";
  if (Serial.available()) {
      delay(10);
      while (Serial.available() > 0) { 
          msg += (char)Serial.read();
      }
      Serial.flush();
  }
}