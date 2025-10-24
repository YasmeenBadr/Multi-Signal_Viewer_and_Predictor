import numpy as np

# Optional backends
try:
    import librosa  # type: ignore
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

try:
    from scipy.signal import resample as _scipy_resample  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _target_len(n_src: int, orig_sr: float, target_sr: float) -> int:
    if orig_sr <= 0 or target_sr <= 0:
        return n_src
    return int(round(n_src * float(target_sr) / float(orig_sr)))


def decimate_with_aliasing(sig: np.ndarray, native_fs: float, target_fs: float, pos_native: int = 0, phase_state: dict | None = None) -> np.ndarray:
    """
    Decimate without anti-aliasing using a persistent phase state (optional).
    - Integer factor: take every k-th sample starting at stored phase p in [0,k-1].
    - Non-integer: use phase-accumulator with initial offset p in [0,factor).
    phase_state is a dict used to persist phase per target_fs. If None, runs statelessly.
    Works for 1D or 2D (samples x channels).
    """
    sig = np.asarray(sig, dtype=np.float32)
    if target_fs <= 0 or target_fs >= native_fs:
        return sig

    factor = float(native_fs) / float(target_fs)
    if factor <= 1.0:
        return sig

    # Optional persistent phase map per target fs key
    phase_map = phase_state if isinstance(phase_state, dict) else {}
    key = int(round(target_fs))
    p_entry = phase_map.get(key, None)

    N = sig.shape[0]
    k = int(np.floor(factor))

    # Integer decimation path
    if abs(factor - k) < 1e-6 and k >= 2:
        if sig.ndim == 1:
            if p_entry is None or not isinstance(p_entry, (int, float)):
                p = int(np.random.randint(0, k))
                phase_map[key] = p
            else:
                p = int(p_entry)
            start = int((min(max(p, 0), k - 1) + (pos_native % k)) % k)
            return sig[start::k]
        else:
            C = sig.shape[1]
            if p_entry is None or not isinstance(p_entry, dict):
                p_dict = {c: int(np.random.randint(0, k)) for c in range(C)}
                phase_map[key] = p_dict
            else:
                p_dict = p_entry
            cols = []
            for c in range(C):
                pc = int(min(max(int(p_dict.get(c, 0)), 0), k - 1))
                pc = int((pc + (pos_native % k)) % k)
                cols.append(sig[pc::k, c])
            maxlen = min(len(col) for col in cols) if cols else 0
            if maxlen == 0:
                return sig[:1, :] if sig.ndim == 2 else sig[:1]
            return np.stack([col[:maxlen] for col in cols], axis=1)

    # Non-integer decimation path using phase accumulator
    if sig.ndim == 1:
        if p_entry is None or not isinstance(p_entry, (int, float)):
            p = float(np.random.uniform(0.0, factor))
            phase_map[key] = p
        else:
            p = float(p_entry)
        p_eff = p + (pos_native % factor)
        nmax = int(np.floor((N - 1 - p_eff) / factor)) + 1 if N > 0 else 0
        if nmax <= 0:
            return sig[:1]
        idx = np.floor(p_eff + np.arange(nmax, dtype=np.float64) * factor).astype(np.int64)
        idx = np.clip(idx, 0, N - 1)
        return sig[idx]
    else:
        C = sig.shape[1]
        if p_entry is None or not isinstance(p_entry, dict):
            p_dict = {c: float(np.random.uniform(0.0, factor)) for c in range(C)}
            phase_map[key] = p_dict
        else:
            p_dict = p_entry
        cols = []
        minlen = None
        for c in range(C):
            pc = float(p_dict.get(c, 0.0))
            pc_eff = pc + (pos_native % factor)
            nmax = int(np.floor((N - 1 - pc_eff) / factor)) + 1 if N > 0 else 0
            if nmax <= 0:
                return sig[:1, :]
            idx = np.floor(pc_eff + np.arange(nmax, dtype=np.float64) * factor).astype(np.int64)
            idx = np.clip(idx, 0, N - 1)
            col = sig[idx, c]
            cols.append(col)
            minlen = len(col) if minlen is None else min(minlen, len(col))
        return np.stack([col[:minlen] for col in cols], axis=1)


def resample_signal(data: np.ndarray, orig_sr: float, target_sr: float, method: str = "scipy", aa: bool = True) -> np.ndarray:
    """
    Unified resampling entry point.
    - If aa is False and target_sr < orig_sr, performs aliasing decimation.
    - Otherwise, uses the chosen backend (scipy or librosa) to resample with anti-aliasing.
    Works for 1D or 2D arrays (samples x channels).
    """
    x = np.asarray(data, dtype=np.float32)
    if target_sr <= 0 or orig_sr <= 0 or int(round(target_sr)) == int(round(orig_sr)):
        return x

    if not aa and target_sr < orig_sr:
        # Stateless aliasing decimation
        return decimate_with_aliasing(x, orig_sr, target_sr)

    # High-quality resampling
    if method == "librosa" and _HAS_LIBROSA:
        if x.ndim == 1:
            return librosa.resample(x, orig_sr=float(orig_sr), target_sr=float(target_sr))
        else:
            # librosa works on mono; process channels independently
            outs = [librosa.resample(x[:, c], orig_sr=float(orig_sr), target_sr=float(target_sr)) for c in range(x.shape[1])]
            minlen = min(len(o) for o in outs)
            return np.stack([o[:minlen] for o in outs], axis=1)

    # Fallback to SciPy if available
    if _HAS_SCIPY:
        n_out = _target_len(x.shape[0], orig_sr, target_sr)
        if x.ndim == 1:
            return _scipy_resample(x, n_out)
        else:
            outs = [_scipy_resample(x[:, c], n_out) for c in range(x.shape[1])]
            return np.stack(outs, axis=1)

    # If no backend available, return input unchanged
    return x
