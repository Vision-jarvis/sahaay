// Sahaay Switch: turns an Arduino (UNO Q, UNO R4, any board with USB serial) into an
// accessibility switch interface for Sahaay on a Snapdragon PC.
//
// Wiring: momentary switches between the pins below and GND (internal pull-ups).
//   D2  switch 1  select      (tap = left click, hold = drag)
//   D3  switch 2  right click
//   D4  switch 3  push-to-talk
//   D5  switch 4  pause / resume head cursor
//   A0/A1 optional joystick (comment out JOYSTICK to disable)
//
// Protocol: one JSON object per line at 115200 baud, see ../README.md.
// Simulate it without hardware at https://wokwi.com (paste this sketch, add 4 pushbuttons).

#define JOYSTICK 1
const uint8_t PINS[4] = {2, 3, 4, 5};
bool last[4] = {false, false, false, false};
unsigned long lastChange[4] = {0, 0, 0, 0};
const unsigned long DEBOUNCE_MS = 25;

void setup() {
  Serial.begin(115200);
  for (uint8_t i = 0; i < 4; i++) pinMode(PINS[i], INPUT_PULLUP);
}

void sendSwitch(uint8_t n, bool down) {
  Serial.print("{\"switch\":"); Serial.print(n);
  Serial.print(",\"state\":\""); Serial.print(down ? "down" : "up"); Serial.println("\"}");
}

void loop() {
  // handshake: Sahaay sends {"hello":"sahaay"} when probing COM ports
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    if (line.indexOf("hello") >= 0) Serial.println("{\"device\":\"sahaay-switch\",\"version\":1}");
  }
  unsigned long now = millis();
  for (uint8_t i = 0; i < 4; i++) {
    bool down = digitalRead(PINS[i]) == LOW;
    if (down != last[i] && now - lastChange[i] > DEBOUNCE_MS) {
      last[i] = down; lastChange[i] = now;
      sendSwitch(i + 1, down);
    }
  }
#ifdef JOYSTICK
  static unsigned long lastJoy = 0;
  if (now - lastJoy > 40) {
    lastJoy = now;
    float x = (analogRead(A0) - 512) / 512.0f;
    float y = (analogRead(A1) - 512) / 512.0f;
    if (fabs(x) > 0.15f || fabs(y) > 0.15f) {
      Serial.print("{\"joy\":["); Serial.print(x, 2); Serial.print(","); Serial.print(y, 2); Serial.println("]}");
    }
  }
#endif
  delay(5);
}
