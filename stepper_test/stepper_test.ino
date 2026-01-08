// ------------------------------
// Stepper motor pins
// ------------------------------
#define STEP_PIN 3    // Connect to STEP on driver
#define DIR_PIN 2     // Connect to DIR on driver

// ------------------------------
// Stepper settings
// ------------------------------
const int stepDelay = 500; // Microseconds between steps (adjust speed)
const unsigned long runTime = 15000; // 10 seconds in milliseconds

void setup() {
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  Serial.begin(9600);
}

void loop() {
  // Turn motor to the right for 10 seconds
  digitalWrite(DIR_PIN, HIGH);
  runMotorFor(runTime);

  // Turn motor to the left for 10 seconds
  digitalWrite(DIR_PIN, LOW);
  runMotorFor(runTime);
}

// Function to run the motor for a certain amount of time
void runMotorFor(unsigned long duration) {
  unsigned long startTime = millis();
  while (millis() - startTime < duration) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(stepDelay);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(stepDelay);
  }
}
