#!/usr/bin/env python3
"""
compare_noarti_vs_raw.py

Sanity-check the noArti pipeline: overlay the calculated (artifact-removed,
low-pass filtered) LFP against the existing raw broadband trace it was derived
from, for a handful of trials across different subjects/sessions.

Each `stim{N}_noArti.mat` file stores two relevant signals:
  - 'data'            : (T, 32) artifact-removed + filtered LFP, all channels
  - 'rawdata'/'data'   : (T, 1)  raw broadband trace for a single channel
                          (whichever channel the MATLAB pipeline last touched
                          before saving — 'rawdata'/'channel' tells us which)
Both are sampled at the same fs, so no resampling is needed to overlay them.
"""

import os, glob, re
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT     = '/Volumes/SanDisk/files/derivative'
OUTPUT_DIR    = os.path.expanduser('~/Desktop/NTS_decoding_results')
STIM_DUR_S    = 5.0     # seconds of stim per block (for trial-window sizing)
PAD_S         = 2.0     # extra context before/after each trial window
RMS_WIN_S     = 0.5     # RMS smoothing window for stim-block detection
N_SUBJECTS    = 3       # how many subjects to sample
N_SESSIONS    = 2       # stim files ("sessions") per subject
N_TRIALS      = 2       # trials per session

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Stim-block detection (same adaptive RMS approach as decode_stim_sham.py) ──
def detect_stim_blocks(stim_sig: np.ndarray, fs: float):
    win = int(RMS_WIN_S * fs)
    rms = np.sqrt(uniform_filter1d(stim_sig.astype(np.float64) ** 2, size=win))
    p10, p90 = np.percentile(rms, 10), np.percentile(rms, 90)
    if p10 < 1e-3:
        threshold = max(0.1, p90 * 0.05)
    elif p90 / (p10 + 1e-9) > 3.0:
        threshold = np.sqrt(p10 * p90)
    else:
        threshold = p10 * 2.0
    binary = (rms > threshold).astype(int)
    d      = np.diff(binary, prepend=0)
    ons    = np.where(d == 1)[0]
    offs   = np.where(d == -1)[0]
    n      = min(len(ons), len(offs))
    return ons[:n], offs[:n]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_noarti_vs_raw(stim_path: str):
    """
    Returns dict: calc (T,) calculated LFP for the raw-backed channel,
    raw (T,) existing raw trace, stim (T,), fs, channel (1-indexed).
    """
    with h5py.File(stim_path, 'r') as f:
        fs      = float(np.array(f['fs']).flat[0])
        channel = int(np.array(f['rawdata']['channel']).flat[0])
        ch_list = np.array(f['ch_list']).squeeze()
        col     = int(np.where(ch_list == channel)[0][0])

        calc = np.array(f['data'][:, col])
        raw  = np.array(f['rawdata']['data']).squeeze()
        stim = np.array(f['stim']).squeeze()
    return dict(calc=calc, raw=raw, stim=stim, fs=fs, channel=channel)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_session(sub_name, stim_num, dat, onsets, out_dir):
    fs       = dat['fs']
    pad_samp = int(PAD_S * fs)
    dur_samp = int(STIM_DUR_S * fs)

    trial_onsets = onsets[:N_TRIALS]
    if len(trial_onsets) == 0:
        print(f'    stim{stim_num}: no stim blocks detected, skipping')
        return

    fig, axes = plt.subplots(len(trial_onsets), 2, figsize=(16, 3.2 * len(trial_onsets)),
                              squeeze=False, gridspec_kw=dict(width_ratios=[3, 1]))
    fig.suptitle(f'{sub_name}  ·  stim{stim_num}  ·  ch{dat["channel"]}  —  '
                 f'noArti (calculated) vs. raw (existing)',
                 fontsize=12, fontweight='bold')

    for i, on in enumerate(trial_onsets):
        lo = max(0, on - pad_samp)
        hi = min(len(dat['calc']), on + dur_samp + pad_samp)
        t    = (np.arange(lo, hi) - on) / fs
        raw  = dat['raw'][lo:hi]
        calc = dat['calc'][lo:hi]
        diff = raw - calc
        corr = np.corrcoef(raw, calc)[0, 1]

        # Overlay
        ax = axes[i, 0]
        ax.plot(t, raw,  color='gray',      lw=0.6, alpha=0.7, label='Raw (existing)')
        ax.plot(t, calc, color='steelblue', lw=1.0,            label='noArti (calculated)')
        ax.axvspan(0, STIM_DUR_S, color='salmon', alpha=0.15, label='Stim ON')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'Trial {i+1}  (onset @ {on/fs:.1f}s)  r={corr:.4f}', fontsize=9)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')

        # Residual (raw - calculated): flat near zero except where artifact
        # removal / filtering actually changed the trace.
        axd = axes[i, 1]
        axd.plot(t, diff, color='indianred', lw=0.6)
        axd.axvspan(0, STIM_DUR_S, color='salmon', alpha=0.15)
        axd.set_title('Raw − Calculated', fontsize=9)
    axes[-1, 0].set_xlabel('Time relative to stim onset (s)')
    axes[-1, 1].set_xlabel('Time relative to stim onset (s)')

    plt.tight_layout()
    out = os.path.join(out_dir, f'{sub_name}_stim{stim_num}_noarti_vs_raw.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    [saved] {out}')


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    subjects = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*')))[:N_SUBJECTS]

    for sub_path in subjects:
        sub_name = os.path.basename(sub_path)
        print(f'\n{"="*65}\n  SUBJECT: {sub_name}\n{"="*65}')

        files     = glob.glob(os.path.join(sub_path, 'stim*_noArti.mat'))
        stim_nums = sorted({
            int(re.search(r'stim(\d+)_', os.path.basename(f)).group(1))
            for f in files
        })[:N_SESSIONS]

        for sn in stim_nums:
            path = os.path.join(sub_path, f'stim{sn}_noArti.mat')
            print(f'  stim{sn}: loading...')
            dat = load_noarti_vs_raw(path)
            ons, _ = detect_stim_blocks(dat['stim'], dat['fs'])
            print(f'  stim{sn}: {len(ons)} stim blocks detected, plotting first {N_TRIALS}')
            plot_session(sub_name, sn, dat, ons, OUTPUT_DIR)

    print('\nDone. Output saved to:', OUTPUT_DIR)
