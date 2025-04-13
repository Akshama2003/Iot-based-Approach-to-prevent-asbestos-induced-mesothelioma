#include <Adafruit_BMP085.h>

Adafruit_BMP085 bmp;

const int buzzerPin = 8; // Connect your buzzer to digital pin 8

void setup() {
  Serial.begin(9600);
  pinMode(buzzerPin, OUTPUT);
  
  if (!bmp.begin()) {
    Serial.println("Could not find a valid BMP085 sensor, check wiring!");
    while (1) {}
  }
}

void loop() {
  float temperature = bmp.readTemperature();

  Serial.print("Body-adjacent Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");

  if (temperature > 37.5) {
    Serial.println("Warning: High body temperature!");
    digitalWrite(buzzerPin, HIGH);  // Turn buzzer ON
  } else if (temperature < 35.0) {
    Serial.println("Warning: Low body temperature!");
    digitalWrite(buzzerPin, HIGH);  // Turn buzzer ON
  } else {
    Serial.println("Temperature within normal range.");
    digitalWrite(buzzerPin, LOW);   // Turn buzzer OFF
  }

  Serial.println();
  delay(1000); // Delay 1 second before next reading
}
