# -*- coding: utf-8 -*-
"""
Script 1: Real-World Data Processing & Hybrid Model Trainer
Dataset : BIDMC PPG and Respiration Dataset (PhysioNet – open access, CC BY 4.0)
Target  : Edge-AI Hypoxia Early Warning System
Run on  : PC / Google Colab
Output  : hypoxia_hybrid_model.tflite  +  scaler_params.json
"""

# ==============================================================================
# 0. IMPORTS & CONFIGURATION
# ==============================================================================
import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
import wfdb
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks as tf_callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

CFG = {
    # Dataset
    'BIDMC_DB'   : 'bidmc/1.0.0',   # used by rdrecord / get_record_list
    'BIDMC_DL'   : 'bidmc',          # used by dl_files (no version suffix)
    'DATA_DIR'   : './bidmc_data',
    'NUM_RECORDS': 53,
    'FS_WAVE'    : 125,   # Hz – waveform (PPG)
    'FS_NUM'     : 1,     # Hz – numerics (HR, SpO2)
    # Windowing
    'WIN_SIZE'   : 30,    # seconds
    'OVERLAP'    : 0.5,
    # Labelling thresholds (clinical)
    'SPO2_CRIT'  : 90,
    'SPO2_WARN'  : 94,
    'HR_LO_CRIT' : 50,
    'HR_HI_CRIT' : 120,
    'HR_LO_WARN' : 55,
    'HR_HI_WARN' : 110,
    'PI_CRIT'    : 0.2,
    'PI_WARN'    : 0.5,
    'LOOK_MIN'   : 120,   # seconds before critical → Warning label starts
    'LOOK_MAX'   : 300,   # seconds before critical → Warning label ends
    # Training
    'BATCH'      : 32,
    'EPOCHS'     : 120,
    'LR'         : 1e-3,
    # Outputs
    'MODEL_OUT'  : 'hypoxia_hybrid_model.tflite',
    'SCALER_OUT' : 'scaler_params.json',
    'PLOT_DIR'   : './training_plots',
}

LABELS      = {0: 'Normal', 1: 'Warning', 2: 'Critical'}
FEAT_NAMES  = ['SpO2', 'HR', 'Temp', 'PI']

# ==============================================================================
# 1. DATASET VERIFICATION (download handled by 00_download_dataset.py)
# ==============================================================================
def verify_dataset():
    """
    Verify that BIDMC dataset files exist in DATA_DIR.
    If missing, instruct user to run 00_download_dataset.py first.
    Returns the number of records found.
    """
    data_dir = CFG['DATA_DIR']
    print(f"\n[1/7] Verifying BIDMC dataset in '{data_dir}' ...")

    if not os.path.isdir(data_dir):
        print(f"     ERROR: Data directory '{data_dir}' not found.")
        print(f"     Please run:  python 00_download_dataset.py")
        sys.exit(1)

    count = 0
    for i in range(1, CFG['NUM_RECORDS'] + 1):
        rname = f"bidmc{i:02d}"
        hea_file = os.path.join(data_dir, f"{rname}.hea")
        if os.path.exists(hea_file):
            count += 1

    if count == 0:
        print(f"     ERROR: No BIDMC records found in '{data_dir}'.")
        print(f"     Please run:  python 00_download_dataset.py")
        sys.exit(1)

    print(f"     Found {count}/{CFG['NUM_RECORDS']} records. Proceeding ...")

# ==============================================================================
# 2. SIGNAL PROCESSING HELPERS
# ==============================================================================
def butter_bandpass(data, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = scipy_signal.butter(order, [lo / nyq, hi / nyq], btype='band')
    return scipy_signal.filtfilt(b, a, data)

def butter_lowpass(data, cutoff, fs, order=4):
    nyq = fs / 2.0
    b, a = scipy_signal.butter(order, cutoff / nyq, btype='low')
    return scipy_signal.filtfilt(b, a, data)

def extract_pi_1hz(ppg_125hz, fs=125):
    """
    Derive Perfusion Index at 1 Hz from raw IR PPG.
    PI = (AC_peak-to-peak / DC_mean) × 100
    AC: bandpass 0.5–4 Hz (pulsatile cardiac component)
    DC: lowpass < 0.08 Hz (slow baseline)
    """
    if len(ppg_125hz) < fs * 2:
        return np.array([])
    ac = butter_bandpass(ppg_125hz, 0.5, 4.0, fs)
    dc = butter_lowpass(ppg_125hz, 0.08, fs)
    pi_out = []
    for i in range(0, len(ppg_125hz) - fs, fs):
        ac_seg = ac[i:i + fs]
        dc_seg = dc[i:i + fs]
        ac_pp  = float(np.ptp(ac_seg))
        dc_mu  = float(np.abs(np.mean(dc_seg)))
        pi_val = (ac_pp / dc_mu * 100.0) if dc_mu > 1e-6 else 0.0
        pi_out.append(np.clip(pi_val, 0.0, 25.0))
    return np.array(pi_out)

def simulate_temperature(spo2_arr, hr_arr):
    """
    Generate a physiologically correlated body-temperature feature (°C).

    Physiological basis:
    • Peripheral vasoconstriction during hypoxia → distal temperature drops
      ~0.05 °C per 1% SpO2 below 95%.
    • Tachycardia/metabolic heat → mild temp rise with HR > 80.
    • Slow autoregressive drift bounded to ±0.4 °C.
    • Gaussian sensor noise σ = 0.08 °C (DS18B20 spec).
    """
    n = len(spo2_arr)
    spo2_effect = -0.05 * np.maximum(0.0, 95.0 - spo2_arr)
    hr_effect   =  0.008 * np.maximum(0.0, hr_arr - 80.0)
    # Autoregressive drift
    drift = np.zeros(n)
    for k in range(1, n):
        drift[k] = 0.995 * drift[k-1] + np.random.normal(0, 0.003)
    drift = np.clip(drift, -0.4, 0.4)
    noise = np.random.normal(0, 0.08, n)
    temp  = 37.0 + spo2_effect + hr_effect + drift + noise
    return np.clip(temp, 35.5, 39.5).astype(np.float32)

# ==============================================================================
# 3. PROCESS ONE RECORD
# ==============================================================================
def process_record(idx):
    """
    Return a DataFrame with columns [SpO2, HR, Temp, PI] aligned at 1 Hz,
    or None on failure.
    Reads from local DATA_DIR (after download) to avoid repeated network calls.
    """
    rname    = f"bidmc{idx:02d}"
    data_dir = CFG['DATA_DIR']
    pn_dir   = CFG['BIDMC_DB']   # fallback: stream from PhysioNet
    try:
        # Prefer local file; fall back to streaming
        local_wf = os.path.join(data_dir, rname + '.hea')
        if os.path.exists(local_wf):
            wf  = wfdb.rdrecord(os.path.join(data_dir, rname))
            num = wfdb.rdrecord(os.path.join(data_dir, rname + 'n'))
        else:
            wf  = wfdb.rdrecord(rname,        pn_dir=pn_dir)
            num = wfdb.rdrecord(rname + 'n',  pn_dir=pn_dir)

        # --- PPG channel (125 Hz) ---
        sig_names = [s.upper() for s in wf.sig_name]
        ppg_idx   = next((i for i, s in enumerate(sig_names)
                          if any(k in s for k in ('PLETH', 'PPG', 'IR'))), None)
        if ppg_idx is None:
            return None
        ppg = wf.p_signal[:, ppg_idx].astype(np.float64)
        ppg = np.where(np.isnan(ppg), np.nanmean(ppg), ppg)

        # --- Numerics (1 Hz) ---
        nnames = [s.upper() for s in num.sig_name]

        def get_ch(keywords):
            for kw in keywords:
                for i, n_ in enumerate(nnames):
                    if kw in n_:
                        return num.p_signal[:, i].astype(np.float32)
            return None

        spo2 = get_ch(['SPO2', 'SAO2', 'O2SAT'])
        hr   = get_ch(['HR', 'PULSE', 'HEART'])   # PULSE is the wrist-HR channel
        if spo2 is None or hr is None:
            return None

        # --- Derived features ---
        pi   = extract_pi_1hz(ppg, fs=CFG['FS_WAVE'])
        n_s  = min(len(spo2), len(hr), len(pi))
        if n_s < CFG['WIN_SIZE'] * 2:
            return None

        spo2 = spo2[:n_s]
        hr   = hr[:n_s]
        pi   = pi[:n_s]
        temp = simulate_temperature(spo2, hr)

        # Clean NaNs — pandas >= 2.0 uses bfill()/ffill() not fillna(method=)
        df = pd.DataFrame({'SpO2': spo2, 'HR': hr, 'Temp': temp, 'PI': pi})
        df = df.interpolate(method='linear').bfill().ffill()
        df = df.dropna()

        # Clip to physiological bounds
        df['SpO2'] = np.clip(df['SpO2'], 70.0, 100.0)
        df['HR']   = np.clip(df['HR'],   20.0, 200.0)
        df['PI']   = np.clip(df['PI'],    0.0,  25.0)
        return df

    except Exception as exc:
        print(f"       [{rname}] skipped: {exc}")
        return None

# ==============================================================================
# 4. LABELLING (look-ahead)
# ==============================================================================
def label_record(df):
    """
    0 = Normal | 1 = Warning (2-5 min pre-hypoxic) | 2 = Critical
    """
    n     = len(df)
    lbls  = np.zeros(n, dtype=np.int8)

    spo2 = df['SpO2'].values
    hr   = df['HR'].values
    pi   = df['PI'].values

    # Pass 1 – mark Critical
    crit = ((spo2 < CFG['SPO2_CRIT']) |
            (hr   < CFG['HR_LO_CRIT']) |
            (hr   > CFG['HR_HI_CRIT']) |
            (pi   < CFG['PI_CRIT']))
    lbls[crit] = 2

    # Pass 2 – look-back from Critical epochs to mark Warning
    lo, hi = CFG['LOOK_MIN'], CFG['LOOK_MAX']
    for t in range(n):
        if lbls[t] == 2:
            start = max(0, t - hi)
            end   = max(0, t - lo)
            for k in range(start, end):
                if lbls[k] == 0:   # don't overwrite Critical
                    lbls[k] = 1

    # Pass 3 – confirm Warning with pre-hypoxic threshold
    warn_cond = ((spo2 >= CFG['SPO2_WARN']) & (spo2 < 95)) | \
                ((hr   >= CFG['HR_LO_WARN']) & (hr < CFG['HR_LO_CRIT'])) | \
                ((hr   >  CFG['HR_HI_WARN']) & (hr < CFG['HR_HI_CRIT'])) | \
                ((pi   >= CFG['PI_CRIT'])     & (pi < CFG['PI_WARN']))
    # Only keep Warning label if physiological warning signs exist
    for t in range(n):
        if lbls[t] == 1 and not warn_cond[t]:
            lbls[t] = 0
    return lbls

# ==============================================================================
# 5. SLIDING WINDOW
# ==============================================================================
def create_windows(df, labels, win=30, overlap=0.5):
    step = int(win * (1 - overlap))
    X, y = [], []
    vals = df[FEAT_NAMES].values.astype(np.float32)
    for i in range(0, len(vals) - win, step):
        X.append(vals[i:i + win])
        # Majority label in window
        seg_lbl = labels[i:i + win]
        counts   = np.bincount(seg_lbl.astype(int), minlength=3)
        y.append(int(np.argmax(counts)))
    return np.array(X), np.array(y)

# ==============================================================================
# 6. DATA AUGMENTATION (minority class)
# ==============================================================================
def augment_windows(X, y, target_ratio=0.30):
    """
    Time-warp + Gaussian noise injection on Warning / Critical windows.
    Ensures each minority class has at least `target_ratio` × majority count.
    """
    counts   = np.bincount(y, minlength=3)
    majority = int(counts.max())
    Xa, ya   = list(X), list(y)
    rng      = np.random.default_rng(0)

    for cls in [1, 2]:   # Warning, Critical
        need = int(majority * target_ratio) - counts[cls]
        if need <= 0:
            continue
        idx_cls = np.where(y == cls)[0]
        if len(idx_cls) == 0:
            continue
        for _ in range(need):
            src = Xa[rng.choice(idx_cls)].copy()
            # 1. Gaussian noise
            src += rng.normal(0, 0.01, src.shape)
            # 2. Amplitude jitter (±5%)
            src *= rng.uniform(0.95, 1.05, (1, src.shape[1]))
            # 3. Time warp: random resample then re-window
            new_len = int(src.shape[0] * rng.uniform(0.90, 1.10))
            stretched = np.array([np.interp(np.linspace(0, 1, src.shape[0]),
                                             np.linspace(0, 1, new_len),
                                             np.interp(np.linspace(0, 1, new_len),
                                                        np.linspace(0, 1, src.shape[0]),
                                                        src[:, c]))
                                   for c in range(src.shape[1])]).T
            Xa.append(stretched)
            ya.append(cls)
    return np.array(Xa, dtype=np.float32), np.array(ya, dtype=np.int32)

# ==============================================================================
# 7. MODEL ARCHITECTURE
# ==============================================================================
def build_hybrid_cnn_lstm(input_shape=(30, 4), n_classes=3):
    """
    Hybrid 1D-CNN + LSTM architecture.
    CNN:  automated feature extractor for PPG pulse morphology.
    LSTM: temporal memory for trend detection leading to hypoxic events.

    TFLite compatibility notes:
    • recurrent_dropout is NOT used — it prevents the TFLite converter
      from fusing LSTM ops into the built-in UnidirectionalSequenceLSTM
      kernel, causing FlexTensorListReserve errors on Raspberry Pi.
    • Standard Dropout layers between LSTM and Dense provide equivalent
      regularisation without breaking TFLite built-in op conversion.
    • unroll=True expands the LSTM time-steps at graph-build time,
      eliminating dynamic TensorList ops that require the Flex delegate.
    """
    inp = layers.Input(shape=input_shape, name='sensor_window')

    # --- CNN branch (feature extraction) ---
    x = layers.Conv1D(64, kernel_size=5, padding='causal',
                      activation='relu', name='conv1')(inp)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling1D(2, name='pool1')(x)

    x = layers.Conv1D(128, kernel_size=3, padding='causal',
                      activation='relu', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling1D(2, name='pool2')(x)

    x = layers.Conv1D(64, kernel_size=3, padding='causal',
                      activation='relu', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)

    # --- LSTM branch (temporal modelling) ---
    # No recurrent_dropout — use inter-layer Dropout instead for TFLite compat.
    # unroll=True ensures no dynamic TensorList ops in the converted graph.
    x = layers.LSTM(64, return_sequences=True, name='lstm1',
                    unroll=True)(x)
    x = layers.Dropout(0.1, name='lstm_drop1')(x)
    x = layers.LSTM(32, return_sequences=False, name='lstm2',
                    unroll=True)(x)

    # --- Classifier head ---
    x = layers.Dense(64, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.35, name='drop1')(x)
    x = layers.Dense(32, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.2,  name='drop2')(x)
    out = layers.Dense(n_classes, activation='softmax', name='risk_prob')(x)

    model = Model(inp, out, name='HypoxiaHybridCNN_LSTM')
    return model

# ==============================================================================
# 8. TRAINING
# ==============================================================================
def train_model(model, X_tr, y_tr, X_val, y_val, cw_dict):
    os.makedirs(CFG['PLOT_DIR'], exist_ok=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CFG['LR']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    cbs = [
        tf_callbacks.EarlyStopping(monitor='val_accuracy', patience=18,
                                   restore_best_weights=True, verbose=1),
        tf_callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                                       patience=8,  min_lr=1e-6, verbose=1),
        tf_callbacks.ModelCheckpoint('best_hypoxia_model.keras',
                                     monitor='val_accuracy',
                                     save_best_only=True, verbose=0),
    ]

    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=CFG['EPOCHS'],
        batch_size=CFG['BATCH'],
        class_weight=cw_dict,
        callbacks=cbs,
        verbose=1,
    )
    return hist

# ==============================================================================
# 9. EVALUATION
# ==============================================================================
def evaluate_and_plot(model, X_te, y_te, history):
    os.makedirs(CFG['PLOT_DIR'], exist_ok=True)
    y_pred_prob = model.predict(X_te, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_te, y_pred,
                                labels=list(LABELS.keys()),
                                target_names=list(LABELS.values()),
                                zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred, labels=list(LABELS.keys()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=LABELS.values(), yticklabels=LABELS.values())
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    axes[1].plot(history.history['accuracy'],     label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].plot(history.history['loss'],         label='Train Loss', linestyle='--')
    axes[1].plot(history.history['val_loss'],     label='Val Loss',   linestyle='--')
    axes[1].set_title('Training History')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CFG['PLOT_DIR'], 'training_summary.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved → {CFG['PLOT_DIR']}/training_summary.png")
    return float(np.mean(y_pred == y_te))

# ==============================================================================
# 10. TFLITE EXPORT — BUILT-IN OPS ONLY (Raspberry Pi compatible)
# ==============================================================================
def quantize_and_export(model, X_repr, scaler_params):
    """
    Convert Keras model to TFLite for Raspberry Pi edge deployment.

    IMPORTANT: Every attempt uses ONLY tf.lite.OpsSet.TFLITE_BUILTINS.
    No SELECT_TF_OPS / Flex delegate is ever enabled.  This guarantees
    the exported .tflite runs on bare tflite_runtime without the Flex
    delegate (which is unavailable on most Raspberry Pi wheels).

    Strategy (cascading fallbacks — all BUILTINS-only):
      1. Full-integer Int8 (weights + activations)     — smallest, fastest
      2. Dynamic-range quantization (int8 weights, float activations)
      3. Float32 (no quantisation, built-in ops only)  — safest fallback
    """
    print("\n[7/7] Converting model to TFLite (BUILTINS-only) ...")

    def rep_dataset():
        for i in range(min(200, len(X_repr))):
            sample = X_repr[i:i+1].astype(np.float32)
            yield [sample]

    tflite_model = None
    quant_type   = None

    # --- Attempt 1: Full-integer Int8 quantization (BUILTINS only) ---
    try:
        print("     Trying full-integer Int8 quantization (BUILTINS) ...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations          = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type   = tf.int8
        converter.inference_output_type  = tf.int8
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        quant_type   = 'full_int8'
        print("     ✓ Full-integer (Int8) quantization successful.")
    except Exception as exc:
        print(f"     ✗ Full-integer Int8 failed: {exc}")

    # --- Attempt 2: Dynamic-range quantisation (BUILTINS only) ---
    if tflite_model is None:
        try:
            print("     Trying dynamic-range quantization (BUILTINS) ...")
            converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
            converter2.optimizations = [tf.lite.Optimize.DEFAULT]
            converter2.representative_dataset = rep_dataset
            converter2.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]
            converter2._experimental_lower_tensor_list_ops = False
            tflite_model = converter2.convert()
            quant_type   = 'dynamic_range_int8'
            print("     ✓ Dynamic-range quantization successful.")
        except Exception as exc:
            print(f"     ✗ Dynamic-range quantization failed: {exc}")

    # --- Attempt 3: Float32 (no quantisation, BUILTINS only) ---
    if tflite_model is None:
        try:
            print("     Trying float32 conversion (BUILTINS only) ...")
            converter3 = tf.lite.TFLiteConverter.from_keras_model(model)
            converter3.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]
            converter3._experimental_lower_tensor_list_ops = False
            tflite_model = converter3.convert()
            quant_type   = 'float32_builtin'
            print("     ✓ Float32 (built-in ops) conversion successful.")
        except Exception as exc:
            print(f"     ✗ Float32 built-in conversion also failed: {exc}")
            print("\n     ERROR: All BUILTINS-only conversion strategies failed.")
            print("     The model architecture contains ops that cannot be")
            print("     mapped to TFLite built-in kernels.  This should not")
            print("     happen with the current CNN+LSTM (unrolled) architecture.")
            print("     Saving .keras model as fallback ...")
            model.save('hypoxia_hybrid_model.keras')
            with open(CFG['SCALER_OUT'], 'w') as f:
                json.dump(scaler_params, f, indent=2)
            print(f"     Scaler saved → {CFG['SCALER_OUT']}")
            return

    # --- Save the BUILTINS-only .tflite ---
    with open(CFG['MODEL_OUT'], 'wb') as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(CFG['MODEL_OUT']) / 1024
    print(f"     Model saved → {CFG['MODEL_OUT']}  ({size_kb:.1f} KB, {quant_type})")
    print(f"     ✓ This model uses ONLY TFLite built-in ops.")
    print(f"       It will run on tflite_runtime without the Flex delegate.")

    # Save scaler params (needed by rpi5_inference.py)
    with open(CFG['SCALER_OUT'], 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"     Scaler saved → {CFG['SCALER_OUT']}")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # Ensure stdout accepts Unicode on Windows
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 65)
    print("  Edge-AI Hypoxia Early Warning System — Model Trainer")
    print("  Dataset : BIDMC PPG & Respiration (PhysioNet, open-access)")
    print("=" * 65)

    # 1. Verify dataset (download via 00_download_dataset.py)
    verify_dataset()

    # 2. Process all records
    print("\n[2/7] Processing BIDMC records ...")
    all_X, all_y = [], []
    ok = 0
    for i in range(1, CFG['NUM_RECORDS'] + 1):
        df = process_record(i)
        if df is None or len(df) < CFG['WIN_SIZE'] * 4:
            continue
        lbl   = label_record(df)
        X_rec, y_rec = create_windows(df, lbl,
                                       win=CFG['WIN_SIZE'],
                                       overlap=CFG['OVERLAP'])
        if len(X_rec) == 0:
            continue
        all_X.append(X_rec)
        all_y.append(y_rec)
        ok += 1
        print(f"     bidmc{i:02d}: {len(X_rec)} windows | "
              f"N={np.sum(y_rec==0)} W={np.sum(y_rec==1)} C={np.sum(y_rec==2)}")

    if ok == 0:
        sys.exit("ERROR: No records processed. Check wfdb / internet connection.")

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int32)
    print(f"\n     Total windows  : {len(X)}")
    print(f"     Class dist     : N={np.sum(y==0)} W={np.sum(y==1)} C={np.sum(y==2)}")

    # 3. Normalise (per-feature min-max across training set)
    print("\n[3/7] Normalising features ...")
    # Physiological clip bounds as hard limits for robust deployment
    FEAT_MIN = np.array([70.0, 20.0, 35.5,  0.0], dtype=np.float32)
    FEAT_MAX = np.array([100.0, 200.0, 39.5, 25.0], dtype=np.float32)
    X = np.clip(X, FEAT_MIN, FEAT_MAX)
    X_norm = (X - FEAT_MIN) / (FEAT_MAX - FEAT_MIN + 1e-8)

    scaler_params = {
        'features': FEAT_NAMES,
        'min': FEAT_MIN.tolist(),
        'max': FEAT_MAX.tolist(),
        'window_size': CFG['WIN_SIZE'],
    }

    # 4. Train / val / test split (stratified)
    print("[4/7] Splitting dataset ...")
    X_tv, X_te, y_tv, y_te = train_test_split(X_norm, y, test_size=0.10,
                                               random_state=42, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tv, y_tv, test_size=0.11,
                                                 random_state=42, stratify=y_tv)

    # 5. Augment minority classes in training set only
    print("[5/7] Augmenting training data ...")
    X_tr, y_tr = augment_windows(X_tr, y_tr, target_ratio=0.40)
    print(f"     After aug: total={len(X_tr)} "
          f"N={np.sum(y_tr==0)} W={np.sum(y_tr==1)} C={np.sum(y_tr==2)}")

    # Class weights for loss
    classes   = np.unique(y_tr)
    cw_vals   = compute_class_weight('balanced', classes=classes, y=y_tr)
    cw_dict   = {int(c): float(w) for c, w in zip(classes, cw_vals)}
    for c in range(3):
        if c not in cw_dict:
            cw_dict[c] = 1.0
    print(f"     Class weights: {cw_dict}")

    # 6. Build + train model
    print("\n[6/7] Building and training Hybrid 1D-CNN + LSTM model ...")
    model   = build_hybrid_cnn_lstm(input_shape=(CFG['WIN_SIZE'], len(FEAT_NAMES)))
    history = train_model(model, X_tr, y_tr, X_val, y_val, cw_dict)
    acc     = evaluate_and_plot(model, X_te, y_te, history)
    print(f"\n     Final test accuracy : {acc*100:.2f}%")

    # 7. Quantise and export
    quantize_and_export(model, X_tr[:300], scaler_params)

    print("\n" + "="*65)
    print("  DONE.  Deploy the following files to Raspberry Pi 5:")
    print(f"  • {CFG['MODEL_OUT']}")
    print(f"  • {CFG['SCALER_OUT']}")
    print("="*65)

if __name__ == '__main__':
    main()
