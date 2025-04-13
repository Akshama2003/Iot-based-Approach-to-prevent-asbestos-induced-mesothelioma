#include <Wire.h>
#include <Adafruit_ADT7410.h>

// Sensor and Pin Definitions
#define BUZZER_PIN 18     // Piezo Buzzer
#define MQ135_PIN 26      // MQ135 for air quality/asbestos dust

// Thresholds
const float TEMP_MIN_THRESHOLD = 20.0;
const float TEMP_MAX_THRESHOLD = 40.0;
const int ASBESTOS_THRESHOLD = 400;
const int ASBESTOS_QUANTITY_LIMIT = 100;

// ADT7410 sensor object
Adafruit_ADT7410 adt = Adafruit_ADT7410();

// Simulated Machine Learning Model
struct MLModel {
  int predict(int mq135Value) {
    return mq135Value > ASBESTOS_THRESHOLD ? 1 : 0;
  }

  int calculateQuantity(int mq135Value) {
    return (mq135Value - 300) / 2;
  }
};

MLModel asbestosModel;

// Simulated µm particle size based on MQ135 value
float simulateParticleSize(int mqValue) {
  // Clamp values for safety
  mqValue = constrain(mqValue, 300, 1023);
  return map(mqValue, 300, 1023, 1, 10); // Returns 1–10 µm range
}

void soundBuzzer(int pattern) {
  if (pattern == 1) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(200);
    digitalWrite(BUZZER_PIN, LOW);
    delay(200);
  } else if (pattern == 2) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(1000);
    digitalWrite(BUZZER_PIN, LOW);
    delay(500);
  }
}

void setup() {
  Serial.begin(115200);

  // Initialize temperature sensor
  if (!adt.begin()) {
    Serial.println("ADT7410 not found. Check wiring!");
    while (1);
  }
  adt.setResolution(ADT7410_16BIT);

  pinMode(MQ135_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.println("System Initialized with ADT7410 & MQ135");
}

void loop() {
  int mq135Value = analogRead(MQ135_PIN);
  float temp = adt.readTempC();
  float particleSize = simulateParticleSize(mq135Value);

  Serial.print("MQ135 Raw Value: "); Serial.print(mq135Value);
  Serial.print(" | Estimated Particle Size: "); Serial.print(particleSize); Serial.println(" µm");

  Serial.print("Temperature: "); Serial.print(temp); Serial.println(" °C");

  int asbestosDetected = asbestosModel.predict(mq135Value);
  int asbestosQuantity = asbestosModel.calculateQuantity(mq135Value);

  bool alertTriggered = false;

  if (asbestosDetected) {
    Serial.print("ALERT: Asbestos Detected! Quantity: ");
    Serial.print(asbestosQuantity);
    Serial.println(" units");
    soundBuzzer(2);
    alertTriggered = true;

    if (asbestosQuantity > ASBESTOS_QUANTITY_LIMIT) {
      Serial.println("CRITICAL: Asbestos Quantity Exceeded Limit!");
    }
  }

  if (temp < TEMP_MIN_THRESHOLD) {
    Serial.println("ALERT: Temperature Too Low!");
    soundBuzzer(1);
    alertTriggered = true;
  } else if (temp > TEMP_MAX_THRESHOLD) {
    Serial.println("ALERT: Temperature Too High!");
    soundBuzzer(1);
    alertTriggered = true;
  }

  if (!alertTriggered) {
    digitalWrite(BUZZER_PIN, LOW);
    Serial.println("All parameters within normal range.");
  }

  Serial.println("--------------------------------------------------");
  delay(2000);
}
