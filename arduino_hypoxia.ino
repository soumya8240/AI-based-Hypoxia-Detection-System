/*
 * ============================================================
 *  Arduino UNO – MAX30102 + DS18B20 + LEDs / Buzzer
 *  Output : SpO2,HR,TempC,PI  (1 Hz CSV on Serial @ 115200)
 *  Input  : 'N' / 'W' / 'C'  (optional override from Raspberry Pi)
 *
 *  LED / BUZZER BEHAVIOUR (responds to Hypoxia category):
 *    Normal   (N) → solid GREEN only,  buzzer silent
 *    Warning  (W) → pulsing YELLOW,    double-beep every 3 s
 *    Critical (C) → solid RED,         rapid alternating buzzer
 *
 *  HYPOXIA THRESHOLDS (onboard – no RPi needed):
 *    SpO2 >= 95                        → Normal
 *    SpO2 90–94  OR  HR < 50 or > 110  → Warning
 *    SpO2 < 90                         → Critical
 *    RPi override ('N'/'W'/'C') always wins when received.
 *
 *  STARTUP SELF-TEST:
 *    On boot, each LED lights for 400 ms then buzzer beeps once.
 *    If any LED/buzzer doesn't respond → wiring problem, not code.
 * ============================================================
 */

#include <Wire.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"
#include <OneWire.h>
#include <DallasTemperature.h>

// ── Pin definitions ─────────────────────────────────────────────
//    GREEN → D5   (any digital pin, no PWM needed)
//    YELLOW→ D6   (MUST be PWM pin for breathing effect)
//    RED   → D7   (any digital pin)
//    BUZZER→ D8   (any digital pin)
//    DS18B20 data → D4
// ────────────────────────────────────────────────────────────────
#define PIN_DS18B20   4
#define PIN_LED_GREEN 5
#define PIN_LED_YELL  6    // PWM pin required for analogWrite()
#define PIN_LED_RED   7
#define PIN_BUZZER    8

// ── Sensor objects ──────────────────────────────────────────────
MAX30105 particleSensor;
OneWire  oneWire(PIN_DS18B20);
DallasTemperature ds18b20(&oneWire);

// ── Algorithm sample buffers ────────────────────────────────────
#define ALGO_SAMPLES 25   // reduced from 50 to save SRAM

uint16_t irBuf[ALGO_SAMPLES];
uint16_t redBuf[ALGO_SAMPLES];

int32_t spo2Val   = 0;
int8_t  validSpo2 = 0;
int32_t hrVal     = 0;
int8_t  validHr   = 0;

uint16_t toAlgo16(uint32_t x) { return (uint16_t)(x >> 2); }

// ── Moving Average ───────────────────────────────────────────────
#define MA_SIZE 4

struct MovAvg {
  uint32_t buf[MA_SIZE];
  uint8_t  idx  = 0;
  uint32_t sum  = 0;
  bool     full = false;

  void push(uint32_t v) {
    sum -= buf[idx];
    buf[idx] = v;
    sum += v;
    if (++idx >= MA_SIZE) { idx = 0; full = true; }
  }
  uint32_t value() const {
    if (full)     return sum / MA_SIZE;
    if (idx == 0) return 0;
    return sum / idx;
  }
};

MovAvg maIR, maRed;

// ── PI calculation ───────────────────────────────────────────────
#define PI_WINDOW 10

uint32_t irWindow[PI_WINDOW];
uint8_t  irWinIdx       = 0;
bool     piWindowFilled = false;

float calcPI(uint32_t dcIR) {
  if (dcIR == 0) return 0.0f;
  uint32_t vMin = irWindow[0], vMax = irWindow[0];
  for (uint8_t i = 1; i < PI_WINDOW; i++) {
    if (irWindow[i] < vMin) vMin = irWindow[i];
    if (irWindow[i] > vMax) vMax = irWindow[i];
  }
  return ((float)(vMax - vMin) / (float)dcIR) * 100.0f;
}

// ── DS18B20 non-blocking ─────────────────────────────────────────
#define DS18B20_RESOLUTION 9    // 9-bit = 94 ms conversion
#define DS18B20_CONV_MS    100UL

float    lastTempC       = -1.0f;
uint32_t tempRequestedAt = 0;

// ── Alert state ──────────────────────────────────────────────────
#define STATE_NORMAL   0
#define STATE_WARNING  1
#define STATE_CRITICAL 2

uint8_t currentState = STATE_NORMAL;
uint8_t prevState    = STATE_NORMAL;

// ── RPi override tracking ─────────────────────────────────────────
// If RPi sends a command, it overrides onboard logic for 5 s.
// After 5 s of silence, onboard SpO2-based logic takes back control.
uint32_t rpiLastCmdMs  = 0;
bool     rpiOverride   = false;
#define  RPI_TIMEOUT_MS  5000UL

// ── Output timing ────────────────────────────────────────────────
uint32_t lastOutput = 0;

// ── WARNING LED vars ─────────────────────────────────────────────
int16_t  yellBrightness = 0;
int8_t   yellDir        = 5;
uint32_t lastPulseMs    = 0;

// WARNING buzzer – 4-phase FSM (non-blocking double-beep every 3 s)
// Phase 0: 3 s idle  1: beep1 ON 80ms  2: gap 80ms  3: beep2 ON 80ms
uint8_t  warnBeepPhase = 0;
uint32_t warnBeepMs    = 0;

// ── CRITICAL vars ────────────────────────────────────────────────
bool     critBuzzerOn = false;
uint32_t critBeepMs   = 0;

// ── Onboard hypoxia thresholds ───────────────────────────────────
uint8_t thresholdState(int32_t spo2, int32_t hr) {
  // If sensor not ready yet, stay normal
  if (spo2 <= 0 || spo2 > 100) return STATE_NORMAL;

  if (spo2 < 90) return STATE_CRITICAL;
  if (spo2 < 95 || hr < 50 || hr > 110) return STATE_WARNING;
  return STATE_NORMAL;
}

// ── Reset actuator timing on every state change ──────────────────
void resetActuatorState() {
  digitalWrite(PIN_LED_GREEN, LOW);
  analogWrite (PIN_LED_YELL,  0);
  digitalWrite(PIN_LED_RED,   LOW);
  digitalWrite(PIN_BUZZER,    LOW);

  yellBrightness = 0;
  yellDir        = 5;
  lastPulseMs    = millis();
  warnBeepPhase  = 0;
  warnBeepMs     = millis();

  critBuzzerOn   = false;
  critBeepMs     = millis();
}

// ── Hardware self-test (runs once at startup) ────────────────────
void selfTest() {
  // GREEN on 400 ms
  digitalWrite(PIN_LED_GREEN, HIGH);
  delay(400);
  digitalWrite(PIN_LED_GREEN, LOW);
  delay(100);

  // YELLOW on 400 ms (full brightness)
  analogWrite(PIN_LED_YELL, 255);
  delay(400);
  analogWrite(PIN_LED_YELL, 0);
  delay(100);

  // RED on 400 ms
  digitalWrite(PIN_LED_RED, HIGH);
  delay(400);
  digitalWrite(PIN_LED_RED, LOW);
  delay(100);

  // BUZZER single beep 200 ms
  digitalWrite(PIN_BUZZER, HIGH);
  delay(200);
  digitalWrite(PIN_BUZZER, LOW);
  delay(100);

  // All three LEDs together 200 ms = test complete
  digitalWrite(PIN_LED_GREEN, HIGH);
  analogWrite (PIN_LED_YELL,  128);
  digitalWrite(PIN_LED_RED,   HIGH);
  delay(200);
  digitalWrite(PIN_LED_GREEN, LOW);
  analogWrite (PIN_LED_YELL,  0);
  digitalWrite(PIN_LED_RED,   LOW);
}

// ── Actuator handler (100% non-blocking) ─────────────────────────
void handleActuators() {
  uint32_t now = millis();

  if (currentState != prevState) {
    resetActuatorState();
    prevState = currentState;
  }

  switch (currentState) {

    // ── NORMAL: solid GREEN, all others off ───────────────────────
    case STATE_NORMAL:
      digitalWrite(PIN_LED_GREEN, HIGH);
      analogWrite (PIN_LED_YELL,  0);
      digitalWrite(PIN_LED_RED,   LOW);
      digitalWrite(PIN_BUZZER,    LOW);
      break;

    // ── WARNING: breathing YELLOW + double-beep every 3 s ─────────
    case STATE_WARNING:
      digitalWrite(PIN_LED_GREEN, LOW);
      digitalWrite(PIN_LED_RED,   LOW);

      // Yellow breathing effect (15 ms tick)
      if (now - lastPulseMs >= 15UL) {
        yellBrightness += yellDir;
        if (yellBrightness >= 250) { yellBrightness = 250; yellDir = -5; }
        if (yellBrightness <=   0) { yellBrightness =   0; yellDir =  5; }
        analogWrite(PIN_LED_YELL, (uint8_t)yellBrightness);
        lastPulseMs = now;
      }

      // Non-blocking double-beep FSM
      switch (warnBeepPhase) {
        case 0: // wait 3 s
          if (now - warnBeepMs >= 3000UL) {
            digitalWrite(PIN_BUZZER, HIGH);
            warnBeepMs = now; warnBeepPhase = 1;
          }
          break;
        case 1: // beep 1 ON – 80 ms
          if (now - warnBeepMs >= 80UL) {
            digitalWrite(PIN_BUZZER, LOW);
            warnBeepMs = now; warnBeepPhase = 2;
          }
          break;
        case 2: // silence gap – 80 ms
          if (now - warnBeepMs >= 80UL) {
            digitalWrite(PIN_BUZZER, HIGH);
            warnBeepMs = now; warnBeepPhase = 3;
          }
          break;
        case 3: // beep 2 ON – 80 ms
          if (now - warnBeepMs >= 80UL) {
            digitalWrite(PIN_BUZZER, LOW);
            warnBeepMs = now; warnBeepPhase = 0;
          }
          break;
      }
      break;

    // ── CRITICAL: solid RED + rapid buzzer 100ms ON / 200ms OFF ───
    case STATE_CRITICAL:
      digitalWrite(PIN_LED_GREEN, LOW);
      analogWrite (PIN_LED_YELL,  0);
      digitalWrite(PIN_LED_RED,   HIGH);

      if (now - critBeepMs >= (critBuzzerOn ? 100UL : 200UL)) {
        critBuzzerOn = !critBuzzerOn;
        digitalWrite(PIN_BUZZER, critBuzzerOn ? HIGH : LOW);
        critBeepMs = now;
      }
      break;
  }
}

// ── SETUP ────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_YELL,  OUTPUT);
  pinMode(PIN_LED_RED,   OUTPUT);
  pinMode(PIN_BUZZER,    OUTPUT);

  // Everything off first
  digitalWrite(PIN_LED_GREEN, LOW);
  analogWrite (PIN_LED_YELL,  0);
  digitalWrite(PIN_LED_RED,   LOW);
  digitalWrite(PIN_BUZZER,    LOW);

  // ── SELF-TEST: lights each LED + beeps buzzer so you can verify wiring ──
  Serial.println(F("SELFTEST:start"));
  selfTest();
  Serial.println(F("SELFTEST:done"));

  // ── DS18B20 ──────────────────────────────────────────────────────
  ds18b20.begin();
  ds18b20.setResolution(DS18B20_RESOLUTION);
  ds18b20.setWaitForConversion(false);
  ds18b20.requestTemperatures();
  tempRequestedAt = millis();

  // ── MAX30102 ──────────────────────────────────────────────────────
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println(F("ERR:MAX30102 not found – check SDA/SCL & 3.3V power"));
    // Rapid red blink to signal error, then soft-reset
    for (uint8_t i = 0; i < 10; i++) {
      digitalWrite(PIN_LED_RED, HIGH); delay(200);
      digitalWrite(PIN_LED_RED, LOW);  delay(200);
    }
    asm volatile ("jmp 0");
  }

  particleSensor.setup(
    /*ledBrightness*/ 0x24,   // ~14 mA
    /*sampleAverage*/ 2,
    /*ledMode*/       2,      // Red + IR
    /*sampleRate*/    100,
    /*pulseWidth*/    411,
    /*adcRange*/      4096
  );
  particleSensor.enableDIETEMPRDY();

  // ── Pre-fill buffers with 10 s timeout ───────────────────────────
  Serial.println(F("INFO:Filling buffers..."));
  uint32_t t0    = millis();
  uint8_t  filled = 0;

  while (filled < ALGO_SAMPLES) {
    if (millis() - t0 > 10000UL) {
      Serial.println(F("WARN:Prefill timeout – check finger on MAX30102"));
      uint16_t fb = (filled > 0) ? irBuf[filled - 1] : 0;
      for (uint8_t k = filled; k < ALGO_SAMPLES; k++)
        irBuf[k] = redBuf[k] = fb;
      break;
    }
    particleSensor.check();
    if (particleSensor.available()) {
      uint32_t ir  = particleSensor.getIR();
      uint32_t red = particleSensor.getRed();
      maIR.push(ir);  maRed.push(red);
      uint32_t sIR = maIR.value(), sRed = maRed.value();
      irBuf[filled]  = toAlgo16(sIR);
      redBuf[filled] = toAlgo16(sRed);
      if (filled < PI_WINDOW) irWindow[filled] = sIR;
      particleSensor.nextSample();
      filled++;
    }
  }

  for (uint8_t i = 0; i < PI_WINDOW; i++)
    if (irWindow[i] == 0) irWindow[i] = maIR.value();

  irWinIdx       = 0;
  piWindowFilled = true;

  resetActuatorState();
  prevState = STATE_NORMAL;

  Serial.println(F("SpO2,HR,Temp,PI"));
}

// ── MAIN LOOP ────────────────────────────────────────────────────
void loop() {
  uint32_t now = millis();

  // ---- 1. Commands from Raspberry Pi (or Serial Monitor for testing) ----
  if (Serial.available() > 0) {
    char cmd = (char)Serial.read();
    if (cmd == 'N' || cmd == 'W' || cmd == 'C') {
      rpiOverride   = true;
      rpiLastCmdMs  = now;
      if      (cmd == 'N') currentState = STATE_NORMAL;
      else if (cmd == 'W') currentState = STATE_WARNING;
      else                 currentState = STATE_CRITICAL;
    }
  }

  // If RPi override has timed out, fall back to onboard threshold logic
  if (rpiOverride && (now - rpiLastCmdMs > RPI_TIMEOUT_MS)) {
    rpiOverride = false;
  }

  // ---- 2. Non-blocking DS18B20 read --------------------------------
  if (now - tempRequestedAt >= DS18B20_CONV_MS) {
    float t = ds18b20.getTempCByIndex(0);
    if (t != DEVICE_DISCONNECTED_C) lastTempC = t;
    ds18b20.requestTemperatures();
    tempRequestedAt = now;
  }

  // ---- 3. Read MAX30102 sample -------------------------------------
  particleSensor.check();
  if (particleSensor.available()) {
    uint32_t rawIR  = particleSensor.getIR();
    uint32_t rawRed = particleSensor.getRed();
    particleSensor.nextSample();

    maIR.push(rawIR);  maRed.push(rawRed);
    uint32_t sIR  = maIR.value();
    uint32_t sRed = maRed.value();

    irWindow[irWinIdx] = sIR;
    if (++irWinIdx >= PI_WINDOW) { irWinIdx = 0; piWindowFilled = true; }

    for (uint8_t i = 0; i < ALGO_SAMPLES - 1; i++) {
      irBuf[i]  = irBuf[i + 1];
      redBuf[i] = redBuf[i + 1];
    }
    irBuf[ALGO_SAMPLES  - 1] = toAlgo16(sIR);
    redBuf[ALGO_SAMPLES - 1] = toAlgo16(sRed);
  }

  // ---- 4. Output every 1 second + update state --------------------
  if (now - lastOutput >= 1000UL) {
    lastOutput = now;

    maxim_heart_rate_and_oxygen_saturation(
      irBuf, ALGO_SAMPLES, redBuf,
      &spo2Val, &validSpo2,
      &hrVal,   &validHr
    );

    int32_t outSpo2 = (validSpo2 && spo2Val > 0 && spo2Val <= 100) ? spo2Val : 0;
    int32_t outHr   = (validHr   && hrVal   > 0 && hrVal   < 250)  ? hrVal   : 0;

    float pi = piWindowFilled ? calcPI(maIR.value()) : 0.0f;
    pi = constrain(pi, 0.0f, 25.0f);

    // ── Onboard threshold logic (active when RPi is not overriding) ──
    if (!rpiOverride) {
      currentState = thresholdState(outSpo2, outHr);
    }

    // Output CSV (matches RPi parse_serial_line format)
    Serial.print(outSpo2);       Serial.print(F(","));
    Serial.print(outHr);         Serial.print(F(","));
    Serial.print(lastTempC, 2);  Serial.print(F(","));
    Serial.println(pi, 3);
  }

  // ---- 5. Drive LEDs and buzzer (non-blocking) --------------------
  handleActuators();
}
