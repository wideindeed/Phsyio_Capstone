"""
Inspect a Keras model file (.h5 or .keras) WITHOUT calling load_model.

Reads architecture from config.json / model_config attrs, and reads raw
weight arrays directly from the HDF5 weight store via h5py. This avoids
all Keras 2 vs Keras 3 deserialization incompatibilities (time_major,
quantization_config, lstm_cell variable mismatches, etc.) since we never
reconstruct the model object at all.

HOW TO RUN:
  1. Edit MODEL_PATH below to point at your model file.
  2. Run this file directly in PyCharm (Run button / Shift+F10).
"""

import os
import io
import json
import zipfile
import numpy as np
import h5py


# =======================================================================
# 👇 EDIT THIS to point at the model you want to inspect
# =======================================================================
MODEL_PATH = "deep_squat_robust.keras"
# =======================================================================


def print_architecture_from_keras(path):
    print("=" * 70)
    print(f"ARCHITECTURE (config.json): {path}")
    print("=" * 70)
    with zipfile.ZipFile(path, 'r') as zf:
        if 'metadata.json' in zf.namelist():
            meta = json.loads(zf.read('metadata.json'))
            print(f"\nSaved with Keras {meta.get('keras_version')} on {meta.get('date_saved')}")

        cfg = json.loads(zf.read('config.json'))
        layers = cfg.get('config', {}).get('layers', [])
        print(f"\n{len(layers)} layers:")
        for layer in layers:
            cls = layer.get('class_name')
            c = layer.get('config', {})
            name = c.get('name')
            extra = ""
            if cls == "InputLayer":
                extra = f"batch_shape={c.get('batch_shape')}"
            elif cls == "Bidirectional":
                inner = c.get('layer', {}).get('config', {})
                extra = f"units={inner.get('units')} return_sequences={inner.get('return_sequences')}"
            elif "Dense" in str(cls):
                extra = f"units={c.get('units')} activation={c.get('activation')}"
            elif cls == "Dropout":
                extra = f"rate={c.get('rate')}"
            print(f"  - {cls:<15s} {name:<20s} {extra}")


def print_architecture_from_h5(path):
    print("=" * 70)
    print(f"ARCHITECTURE (HDF5 attrs): {path}")
    print("=" * 70)
    with h5py.File(path, 'r') as f:
        if 'model_config' in f.attrs:
            val = f.attrs['model_config']
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            cfg = json.loads(val)
            layers = cfg.get('config', {}).get('layers', [])
            print(f"\n{len(layers)} layers:")
            for layer in layers:
                cls = layer.get('class_name')
                c = layer.get('config', {})
                name = c.get('name')
                extra = ""
                if cls == "InputLayer":
                    extra = f"batch_shape={c.get('batch_shape')}"
                elif cls == "Bidirectional":
                    inner = c.get('layer', {}).get('config', {})
                    extra = f"units={inner.get('units')} return_sequences={inner.get('return_sequences')}"
                elif "Dense" in str(cls):
                    extra = f"units={c.get('units')} activation={c.get('activation')}"
                elif cls == "Dropout":
                    extra = f"rate={c.get('rate')}"
                print(f"  - {cls:<15s} {name:<20s} {extra}")


def lstm_bias_verdict(bias):
    """unit_forget_bias=True LSTM init => 4 gate blocks with means [0,1,0,0]."""
    n = bias.shape[0] // 4
    blocks = [bias[i*n:(i+1)*n] for i in range(4)]
    means = np.array([b.mean() for b in blocks])
    target = np.array([0.0, 1.0, 0.0, 0.0])
    dist = np.abs(means - target).mean()
    if dist < 0.02:
        return f"LIKELY UNTRAINED (gate-block means {np.round(means,4)} ~= init [0,1,0,0])"
    else:
        return f"looks trained (gate-block means {np.round(means,4)})"


def dense_kernel_verdict(kernel):
    """Compare std to theoretical GlorotUniform init std for this shape."""
    fan_in, fan_out = kernel.shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    init_std = limit / np.sqrt(3.0)
    actual_std = kernel.std()
    ratio = actual_std / init_std if init_std > 0 else float('nan')
    if 0.7 < ratio < 1.3:
        return f"LIKELY UNTRAINED (std={actual_std:.4f} ~= init std={init_std:.4f}, ratio={ratio:.2f})"
    else:
        return f"looks trained (std={actual_std:.4f} vs init std={init_std:.4f}, ratio={ratio:.2f})"


def dump_weights(h5file):
    """Walk every dataset in an open h5py.File and print stats + heuristics."""
    print("\n--- Weight arrays ---")

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        arr = obj[()]
        if arr.size == 0:
            return
        print(f"\n  {name}")
        print(f"    shape={arr.shape} dtype={arr.dtype} "
              f"mean={arr.mean():+.5f} std={arr.std():.5f} "
              f"min={arr.min():+.5f} max={arr.max():+.5f}")

        lower = name.lower()
        if "bias" in lower and arr.ndim == 1 and arr.shape[0] % 4 == 0 and "lstm" in lower:
            print(f"    -> {lstm_bias_verdict(arr)}")
        elif "kernel" in lower and arr.ndim == 2 and "recurrent" not in lower and "dense" in lower:
            print(f"    -> {dense_kernel_verdict(arr)}")

    h5file.visititems(visit)


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ File not found: {os.path.abspath(MODEL_PATH)}")
        print("   Edit MODEL_PATH at the top of this script.")
    else:
        ext = os.path.splitext(MODEL_PATH)[1].lower()

        if ext == ".keras":
            print_architecture_from_keras(MODEL_PATH)
            with zipfile.ZipFile(MODEL_PATH, 'r') as zf:
                weights_bytes = zf.read('model.weights.h5')
            with h5py.File(io.BytesIO(weights_bytes), 'r') as f:
                dump_weights(f)

        elif ext == ".h5":
            print_architecture_from_h5(MODEL_PATH)
            with h5py.File(MODEL_PATH, 'r') as f:
                root = f['model_weights'] if 'model_weights' in f else f
                dump_weights(root)

        else:
            print(f"Unrecognized extension '{ext}'")
