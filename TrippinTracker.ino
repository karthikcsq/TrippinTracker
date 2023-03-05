#include <NewPing.h>

#define TRIGGER_PIN_right  12  // Arduino pin tied to trigger pin on the ultrasonic sensor.
#define ECHO_PIN_right     11  // Arduino pin tied to echo pin on the ultrasonic sensor.
#define TRIGGER_PIN_left  10  // Arduino pin tied to trigger pin on the ultrasonic sensor.
#define ECHO_PIN_left     9  // Arduino pin tied to echo pin on the ultrasonic sensor.
#define MAX_DISTANCE 200 // Maximum distance we want to ping for (in centimeters). Maximum sensor distance is rated at 400-500cm.

NewPing sonar_R(TRIGGER_PIN_right, ECHO_PIN_right, MAX_DISTANCE); // NewPing setup of pins and maximum distance.
NewPing sonar_L(TRIGGER_PIN_left, ECHO_PIN_left, MAX_DISTANCE); // NewPing setup of pins and maximum distance.

int thresh = 6;

// unsigned int prevRight = 0; // Send ping, get ping time in microseconds (uS).
// unsigned int prevLeft = 0;

void setup() {
  Serial.begin(9600); // Open serial monitor at 115200 baud to see ping results.
}

void loop() {
  delay(250);  // Wait 500ms between pings (about 2 pings/sec). 29ms should be the shortest delay between pings.
  unsigned int uSR = sonar_R.ping(); // Send ping, get ping time in microseconds (uS).
  unsigned int uSL = sonar_L.ping();

  bool rightStep = (uSR / US_ROUNDTRIP_CM)>thresh;
  bool leftStep = (uSL / US_ROUNDTRIP_CM)>thresh;
  unsigned int rightVal = (uSR / US_ROUNDTRIP_CM);
  unsigned int leftVal = (uSL / US_ROUNDTRIP_CM);
  // Serial.print("Ping: ");


  Serial.print(leftStep); // Convert ping time to distance and print result (0 = outside set distance range, no ping echo)
  Serial.print(" ");
  Serial.print(rightStep);
  Serial.print("    ");
  Serial.print(leftVal);
  Serial.print(" ");
  Serial.println(rightVal);


  // Serial.println("cm");
}