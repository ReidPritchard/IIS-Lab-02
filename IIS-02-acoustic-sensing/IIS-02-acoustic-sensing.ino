#define BUZZER_PIN 27

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
}

void loop() {
  int hrtz = 440;
  tone(BUZZER_PIN, hrtz);
  delay(500);
  noTone(BUZZER_PIN);
  delay(5000);
}
