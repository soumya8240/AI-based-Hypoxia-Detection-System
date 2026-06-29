#!/usr/bin/env python3
"""
Script 3 : Raspberry Pi 5 – Edge Inference, Kalman Filtering & Cloud Upload
Project  : Edge-AI Hypoxia Early Warning System

Hardware USB: Arduino connected to Raspberry Pi 5 via USB-A to USB-B cable.
              Arduino UNO appears as /dev/ttyACM0 (or /dev/ttyUSB0 for
              CH340-based clones).  No GPIO wiring or voltage divider needed.
Baud rate: 115200

Serial from Arduino : "SpO2,HR,Temp,PI\\n"  (1 Hz)
Serial to Arduino   : 'N', 'W', or 'C'      (after each inference)

Cloud   : ThingSpeak  Channel 3375745
Requirements (Raspberry Pi):
    pip install pyserial tflite-runtime numpy requests
    (tflite-runtime wheels: https://github.com/google-coral/pycoral/releases)
    Permissions: sudo usermod -aG dialout $USER  (then log out/in)
"""

import os
import sys
import csv
import json
import time
import logging
import threading
import traceback
from collections import deque
from datetime import datetime

import numpy as np
import requests
import serial

# tflite_runtime is the lightweight runtime for Raspberry Pi
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback: full TensorFlow (slower on Pi, but works for testing)
    import tensorflow as tf
    tflite = tf.lite

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Candidate USB serial ports (tried in order).
# Arduino UNO (ATmega16U2 USB chip) → /dev/ttyACM0
# Arduino UNO clones (CH340 chip)   → /dev/ttyUSB0
USB_PORT_CANDIDATES = ['/dev/ttyACM0', '/dev/ttyUSB0',
                       '/dev/ttyACM1', '/dev/ttyUSB1']

CFG = {
    # Serial – USB cable (Arduino → Raspberry Pi 5 via USB-A to USB-B)
    'SERIAL_PORT'    : None,             # auto-detected from USB_PORT_CANDIDATES
    'BAUD_RATE'      : 115200,
    'SERIAL_TIMEOUT' : 2.0,              # seconds
    # Model
    'MODEL_PATH'     : 'hypoxia_hybrid_model.tflite',
    'SCALER_PATH'    : 'scaler_params.json',
    'WIN_SIZE'       : 30,               # must match training window
    'N_FEATURES'     : 4,                # SpO2, HR, Temp, PI
    # Kalman filter noise parameters (tuned per-channel)
    # Lower Q → slower adaptation (smoother), Higher R → less trust in measurement
    'KALMAN_Q'  : [0.05, 0.5,  0.005, 0.02 ],  # [SpO2, HR, Temp, PI]
    'KALMAN_R'  : [0.5,  2.0,  0.01,  0.1  ],  # [SpO2, HR, Temp, PI]
    # Output
    'CSV_PATH'       : 'vitals.csv',
    # ThingSpeak
    'TS_API_KEY'     : '06NEPCTU2C2QVQUH',
    'TS_CHANNEL_ID'  : '3375745',
    'TS_URL'         : 'https://api.thingspeak.com/update',
    'TS_INTERVAL_S'  : 15,              # push every N seconds
    'TS_MAX_RETRIES' : 3,
    'TS_RETRY_DELAY' : 5,              # seconds between retries
    # Inference
    'INFERENCE_HZ'   : 1,              # run inference every second
    'LABELS'         : ['Normal', 'Warning', 'Critical'],
    'CMD_MAP'        : {0: b'N', 1: b'W', 2: b'C'},
}

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('HypoxiaEdge')

# ==============================================================================
# 1. KALMAN FILTER (1-D scalar, per channel)
# ==============================================================================
class ScalarKalman:
    """
    Minimal 1-D Kalman filter for scalar physiological signals.

    Eliminates motion artifacts and sensor noise that the MAX30102
    is highly susceptible to (motion = high-frequency spikes in PI / HR).

    State transition : x_k = x_{k-1}  (constant model; signal changes slowly)
    Observation      : z_k = x_k + v_k  (v_k ~ N(0, R))
    Process noise    : w_k ~ N(0, Q)

    Tuning guide:
      • Q (process noise) : increase to track fast physiological changes.
      • R (meas. noise)   : increase if raw signal is very noisy.
    """
    def __init__(self, Q: float, R: float):
        self.Q = float(Q)
        self.R = float(R)
        self.x = None          # state estimate (initialised on first sample)
        self.P = 1.0           # error covariance

    def update(self, z: float) -> float:
        if self.x is None:
            self.x = z
            return z
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q
        # Update (Kalman gain)
        K      = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (z - x_pred)
        self.P = (1.0 - K) * P_pred
        return float(self.x)

# ==============================================================================
# 2. MODEL LOADER
# ==============================================================================
class HypoxiaInferenceEngine:
    """Wraps TFLite interpreter with input normalisation."""

    def __init__(self, model_path: str, scaler_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler params not found: {scaler_path}")

        # Load scaler parameters
        with open(scaler_path, 'r') as f:
            sp = json.load(f)
        self.feat_min = np.array(sp['min'],  dtype=np.float32)
        self.feat_max = np.array(sp['max'],  dtype=np.float32)
        self.win_size = int(sp['window_size'])

        # Load TFLite model
        self.interp = tflite.Interpreter(model_path=model_path,
                                         num_threads=4)
        self.interp.allocate_tensors()
        self.in_idx  = self.interp.get_input_details()[0]['index']
        self.out_idx = self.interp.get_output_details()[0]['index']

        in_details  = self.interp.get_input_details()[0]
        out_details = self.interp.get_output_details()[0]
        self.input_is_int8  = (in_details['dtype']  == np.int8)
        self.output_is_int8 = (out_details['dtype'] == np.int8)

        # Quantisation scale/zero-point for int8 models
        if self.input_is_int8:
            self.in_scale, self.in_zp = in_details['quantization']
        if self.output_is_int8:
            self.out_scale, self.out_zp = out_details['quantization']

        log.info("Model loaded. Input: %s | Output: %s",
                 in_details['dtype'], out_details['dtype'])

    def predict(self, window: np.ndarray):
        """
        window : ndarray shape (WIN_SIZE, 4)  – raw Kalman-filtered values
        Returns: (class_idx, probabilities_array)
        """
        # Normalise to [0, 1]
        win_norm = (window - self.feat_min) / (self.feat_max - self.feat_min + 1e-8)
        win_norm = np.clip(win_norm, 0.0, 1.0).astype(np.float32)
        inp      = win_norm[np.newaxis, ...]   # (1, WIN_SIZE, 4)

        if self.input_is_int8:
            inp = (inp / self.in_scale + self.in_zp).astype(np.int8)

        self.interp.set_tensor(self.in_idx, inp)
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out_idx)[0]  # shape (3,)

        if self.output_is_int8:
            probs = (raw.astype(np.float32) - self.out_zp) * self.out_scale
        else:
            probs = raw.astype(np.float32)

        probs = np.clip(probs, 0.0, 1.0)
        probs /= (probs.sum() + 1e-8)   # re-normalise after quant rounding
        return int(np.argmax(probs)), probs

# ==============================================================================
# 3. SERIAL READER
# ==============================================================================
def parse_serial_line(line: str):
    """
    Parse "SpO2,HR,Temp,PI\\n" from Arduino.
    Returns (spo2, hr, temp, pi) floats or None on malformed input.
    """
    try:
        parts = line.strip().split(',')
        if len(parts) != 4:
            return None
        spo2, hr, temp, pi = map(float, parts)
        # Sanity check – discard obviously bad readings
        if spo2 == 0 and hr == 0:   # sensor not ready
            return None
        spo2 = np.clip(spo2, 70.0,  100.0)
        hr   = np.clip(hr,   20.0,  200.0)
        temp = np.clip(temp, 30.0,   42.0)
        pi   = np.clip(pi,    0.0,   25.0)
        return float(spo2), float(hr), float(temp), float(pi)
    except (ValueError, AttributeError):
        return None

# ==============================================================================
# 4. CSV LOGGER
# ==============================================================================
class VitalsLogger:
    COLS = ['timestamp', 'spo2_raw', 'hr_raw', 'temp_raw', 'pi_raw',
            'spo2_k', 'hr_k', 'temp_k', 'pi_k',
            'prediction', 'confidence_pct']

    def __init__(self, path: str):
        self.path  = path
        self._lock = threading.Lock()
        write_header = not os.path.exists(path)
        self._fh  = open(path, 'a', newline='')
        self._csv = csv.DictWriter(self._fh, fieldnames=self.COLS)
        if write_header:
            self._csv.writeheader()
            self._fh.flush()
        log.info("CSV logger → %s", os.path.abspath(path))

    def write(self, raw, kalman, pred_label, confidence):
        row = {
            'timestamp'     : datetime.now().isoformat(timespec='seconds'),
            'spo2_raw'      : f'{raw[0]:.1f}',
            'hr_raw'        : f'{raw[1]:.1f}',
            'temp_raw'      : f'{raw[2]:.2f}',
            'pi_raw'        : f'{raw[3]:.3f}',
            'spo2_k'        : f'{kalman[0]:.1f}',
            'hr_k'          : f'{kalman[1]:.1f}',
            'temp_k'        : f'{kalman[2]:.2f}',
            'pi_k'          : f'{kalman[3]:.3f}',
            'prediction'    : pred_label,
            'confidence_pct': f'{confidence*100:.1f}',
        }
        with self._lock:
            self._csv.writerow(row)
            self._fh.flush()

    def close(self):
        self._fh.close()

# ==============================================================================
# 5. THINGSPEAK UPLOADER  (runs in a background thread every 15 s)
# ==============================================================================
class ThingSpeakUploader:
    """
    Thread-safe ThingSpeak uploader with exponential backoff on failure.
    Field mapping:
      field1 = SpO2 (Kalman), field2 = HR (Kalman)
      field3 = Temperature,   field4 = PI (Kalman)
      field5 = Prediction (0/1/2), field6 = Confidence (%)
    """
    def __init__(self, api_key, channel_id, interval_s=15):
        self.api_key    = api_key
        self.channel_id = channel_id
        self.interval   = interval_s
        self._latest    = None
        self._lock      = threading.Lock()
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True,
                                           name='ThingSpeakUploader')
        self._thread.start()
        log.info("ThingSpeak uploader started (channel %s, every %ds)",
                 channel_id, interval_s)

    def update(self, kalman_vals, pred_idx, confidence):
        """Called from main thread; stores latest data for background push."""
        with self._lock:
            self._latest = (kalman_vals, pred_idx, confidence)

    def _push(self, kalman_vals, pred_idx, confidence):
        payload = {
            'api_key': self.api_key,
            'field1' : round(float(kalman_vals[0]), 1),   # SpO2
            'field2' : round(float(kalman_vals[1]), 1),   # HR
            'field3' : round(float(kalman_vals[2]), 2),   # Temp
            'field4' : round(float(kalman_vals[3]), 3),   # PI
            'field5' : int(pred_idx),                     # Prediction class
            'field6' : round(float(confidence * 100), 1), # Confidence %
        }
        for attempt in range(1, CFG['TS_MAX_RETRIES'] + 1):
            try:
                r = requests.post(CFG['TS_URL'], data=payload, timeout=10)
                if r.status_code == 200 and r.text.strip() != '0':
                    log.info("[ThingSpeak] ✓ Entry %s | SpO2=%.1f HR=%.1f "
                             "Temp=%.2f PI=%.3f → %s (%.0f%%)",
                             r.text.strip(),
                             kalman_vals[0], kalman_vals[1],
                             kalman_vals[2], kalman_vals[3],
                             CFG['LABELS'][pred_idx], confidence * 100)
                    return
                else:
                    log.warning("[ThingSpeak] Bad response: %s", r.text.strip())
            except requests.exceptions.RequestException as exc:
                log.warning("[ThingSpeak] Attempt %d/%d failed: %s",
                            attempt, CFG['TS_MAX_RETRIES'], exc)
            if attempt < CFG['TS_MAX_RETRIES']:
                time.sleep(CFG['TS_RETRY_DELAY'] * attempt)   # backoff
        log.error("[ThingSpeak] All %d attempts failed. Data not pushed.",
                  CFG['TS_MAX_RETRIES'])

    def _run(self):
        while not self._stop.wait(timeout=self.interval):
            with self._lock:
                data = self._latest
            if data is not None:
                self._push(*data)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

# ==============================================================================
# 6.  MAIN INFERENCE LOOP
# ==============================================================================
def detect_usb_port() -> str:
    """
    Return the first available USB serial port from USB_PORT_CANDIDATES.
    Raises RuntimeError if none are found.
    """
    import glob
    for candidate in USB_PORT_CANDIDATES:
        # Also expand wildcards like /dev/ttyACM* in case index differs
        matches = glob.glob(candidate)
        for port in matches:
            try:
                # Try opening briefly to confirm the device is accessible
                test = serial.Serial(port, CFG['BAUD_RATE'],
                                     timeout=0.5)
                test.close()
                log.info("Arduino USB port detected: %s", port)
                return port
            except (serial.SerialException, OSError):
                continue
    raise RuntimeError(
        "No Arduino USB port found. Tried: " + str(USB_PORT_CANDIDATES) +
        "\nEnsure the USB cable is connected and you are in the 'dialout' group:"
        "\n  sudo usermod -aG dialout $USER  (then log out and back in)"
    )


def main():
    log.info("="*60)
    log.info("  Edge-AI Hypoxia Inference Engine – Raspberry Pi 5")
    log.info("  Communication : USB cable (Arduino → RPi 5)")
    log.info("="*60)

    # ── Load model ──────────────────────────────────────────────
    engine = HypoxiaInferenceEngine(CFG['MODEL_PATH'], CFG['SCALER_PATH'])
    log.info("Model ready. Window size = %d samples.", engine.win_size)

    # ── 4× Kalman filters (SpO2, HR, Temp, PI) ─────────────────
    filters = [ScalarKalman(Q=CFG['KALMAN_Q'][i],
                            R=CFG['KALMAN_R'][i]) for i in range(4)]

    # ── Sliding window buffer ────────────────────────────────────
    window = deque(maxlen=engine.win_size)
    # Seed with safe physiological defaults to allow early inference
    default_sample = [98.0, 72.0, 37.0, 1.5]
    for _ in range(engine.win_size):
        window.append(default_sample.copy())

    # ── CSV logger ───────────────────────────────────────────────
    logger = VitalsLogger(CFG['CSV_PATH'])

    # ── ThingSpeak uploader (background thread) ─────────────────
    ts_uploader = ThingSpeakUploader(CFG['TS_API_KEY'],
                                     CFG['TS_CHANNEL_ID'],
                                     CFG['TS_INTERVAL_S'])

    # ── Auto-detect Arduino USB port ─────────────────────────────
    if CFG['SERIAL_PORT'] is None:
        while True:
            try:
                CFG['SERIAL_PORT'] = detect_usb_port()
                break
            except RuntimeError as exc:
                log.error("%s\nRetrying in 5 s …", exc)
                time.sleep(5)

    log.info("Serial port : %s @ %d baud", CFG['SERIAL_PORT'], CFG['BAUD_RATE'])

    # ── Open serial port ─────────────────────────────────────────
    ser = None
    while ser is None:
        try:
            ser = serial.Serial(CFG['SERIAL_PORT'], CFG['BAUD_RATE'],
                                timeout=CFG['SERIAL_TIMEOUT'])
            log.info("USB serial port %s opened successfully.", CFG['SERIAL_PORT'])
        except serial.SerialException as exc:
            log.error("Cannot open serial: %s. Retrying in 5s …", exc)
            time.sleep(5)

    log.info("Waiting for sensor data …")
    prev_pred  = 0
    loop_count = 0

    try:
        while True:
            # ── Read one line from Arduino ───────────────────────
            try:
                raw_line = ser.readline().decode('ascii', errors='ignore')
            except serial.SerialException as exc:
                log.error("Serial read error: %s. Attempting reconnect …", exc)
                time.sleep(2)
                try:
                    ser.close()
                    ser.open()
                except Exception:
                    pass
                continue

            parsed = parse_serial_line(raw_line)
            if parsed is None:
                continue

            raw_spo2, raw_hr, raw_temp, raw_pi = parsed
            raw_vec = [raw_spo2, raw_hr, raw_temp, raw_pi]

            # ── Apply Kalman filters ─────────────────────────────
            kalman_vec = [filters[i].update(raw_vec[i]) for i in range(4)]

            # ── Update sliding window ────────────────────────────
            window.append(kalman_vec)

            # ── Run inference ────────────────────────────────────
            win_array  = np.array(window, dtype=np.float32)  # (30, 4)
            pred_idx, probs = engine.predict(win_array)
            confidence  = float(probs[pred_idx])
            pred_label  = CFG['LABELS'][pred_idx]

            # ── Print inference result ───────────────────────────
            loop_count += 1
            log.info(
                "#%04d | SpO2=%5.1f→%5.1f  HR=%4.0f→%4.0f  "
                "Temp=%5.2f→%5.2f  PI=%5.3f→%5.3f | "
                "▶ %-8s (N=%.2f W=%.2f C=%.2f)",
                loop_count,
                raw_spo2,   kalman_vec[0],
                raw_hr,     kalman_vec[1],
                raw_temp,   kalman_vec[2],
                raw_pi,     kalman_vec[3],
                pred_label,
                probs[0], probs[1], probs[2],
            )

            # ── Send command to Arduino ──────────────────────────
            cmd = CFG['CMD_MAP'][pred_idx]
            try:
                ser.write(cmd)
            except serial.SerialException as exc:
                log.warning("Serial write error: %s", exc)

            # ── Log to CSV ───────────────────────────────────────
            logger.write(raw_vec, kalman_vec, pred_label, confidence)

            # ── Update ThingSpeak buffer ─────────────────────────
            ts_uploader.update(kalman_vec, pred_idx, confidence)

    except KeyboardInterrupt:
        log.info("\nShutdown requested. Closing resources …")

    finally:
        ts_uploader.stop()
        logger.close()
        if ser and ser.is_open:
            ser.write(b'N')    # reset Arduino to Normal on exit
            ser.close()
        log.info("Edge inference engine stopped cleanly.")

# ==============================================================================
if __name__ == '__main__':
    main()
