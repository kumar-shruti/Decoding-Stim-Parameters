#!/usr/bin/env python3
"""
decode_stim_sham.py

Decode stimulation vs. sham using multiunit spiking activity (MUA) and LFP.
  - 5 subjects from /Volumes/SanDisk/files/derivative/
  - Stim blocks detected with adaptive RMS thresholding
  - Feature extraction: mean firing rate (MUA) + RMS amplitude (LFP) per channel
  - Classifier: SVM
  - Decoding is done per subject: trials are pooled across all of that
    subject's sessions, and cross-validated with trial-level Leave-One-Out
    (each fold leaves out one trial, not one session).
  - Three sham-block definitions are each decoded separately:
      1. pre       — 5 s immediately before each stim onset
      2. between   — 5 s centered in the gap between consecutive stim blocks
      3. baseline  — the leading baseline (recording start -> first stim
                     onset) chopped into consecutive 5 s increments
  - Per-subject output: activity bar chart + a 2 (LFP/MUA) x 3 (sham
    definition) grid of confusion matrices, each cell normalized to show
    the percentage of true-label trials assigned to each predicted label.
"""

import os, glob, re, warnings
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT    = '/Volumes/SanDisk/files/derivative'
OUTPUT_DIR   = os.path.expanduser('~/Desktop/NTS_decoding_results')
STIM_DUR_S   = 5.0     # seconds of stim to extract per block
SHAM_WIN_S   = 5.0     # seconds per sham window (all three definitions)
RMS_WIN_S    = 0.5     # RMS smoothing window
N_CH         = 32
RUN_DECODE   = False   # set True to also run LOO decoding + confusion-matrix plots

MODALITIES = ['LFP', 'MUA']
SHAM_DEFINITIONS = [
    ('pre',      '5s Pre-Stim'),
    ('between',  '5s Inter-Stim'),
    ('baseline', '5s Baseline Chop'),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_stim_file(sub_dir: str, stim_num: int):
    """
    Load LFP (noArti) and MUA (MUA_100 > mua_50 fallback) for one stim file.
    Returns dict with keys: lfp, mua, stim, time, fs  –– or None if missing.
    """
    base     = os.path.join(sub_dir, f'stim{stim_num}')
    lfp_path = base + '_noArti.mat'
    if not os.path.exists(lfp_path):
        return None

    with h5py.File(lfp_path, 'r') as f:
        lfp      = np.array(f['data']).squeeze()    # (T, 32)
        stim_sig = np.array(f['stim']).squeeze()    # (T,)
        time     = np.array(f['time']).squeeze()    # (T,)
        fs       = float(np.array(f['fs']).flat[0])

    # MUA: prefer explicit MUA file; fall back to spike_train inside LFP file
    mua = None
    for suffix in ('_MUA_100.mat', '_mua_50.mat', '_MUA_50.mat'):
        mua_path = base + suffix
        if os.path.exists(mua_path):
            with h5py.File(mua_path, 'r') as f:
                mua = np.array(f['spike_train'])    # (T', 32)
            break

    # Align lengths if MUA differs
    T = lfp.shape[0]
    if mua is not None and mua.shape[0] != T:
        n = min(mua.shape[0], T)
        lfp      = lfp[:n]
        stim_sig = stim_sig[:n]
        time     = time[:n]
        mua      = mua[:n]

    return dict(lfp=lfp, mua=mua, stim=stim_sig, time=time, fs=fs)


# ── Stim-block detection ──────────────────────────────────────────────────────
def detect_stim_blocks(stim_sig: np.ndarray, fs: float):
    """
    Adaptive RMS-envelope threshold to locate stim ON blocks.
    Works for both small-amplitude (sub-1/2/5, |stim|~3 V) and
    large-amplitude (sub-3/4, |stim|~1e6 µV pulse transients).

    Cast to float64 before squaring to avoid float32 precision loss.
    Returns (onsets_samp, offsets_samp), both ascending / index-aligned.
    """
    win = int(RMS_WIN_S * fs)
    rms = np.sqrt(uniform_filter1d(stim_sig.astype(np.float64) ** 2, size=win))

    p10 = np.percentile(rms, 10)
    p90 = np.percentile(rms, 90)

    if p10 < 1e-3:
        # Near-zero baseline: fixed or fraction-of-peak threshold
        threshold = max(0.1, p90 * 0.05)
    elif p90 / (p10 + 1e-9) > 3.0:
        # Bimodal – geometric mean of low and high percentiles
        threshold = np.sqrt(p10 * p90)
    else:
        threshold = p10 * 2.0

    binary = (rms > threshold).astype(int)
    d      = np.diff(binary, prepend=0)
    ons    = np.where(d ==  1)[0]
    offs   = np.where(d == -1)[0]
    n      = min(len(ons), len(offs))
    return ons[:n], offs[:n]


# ── Segment extraction ────────────────────────────────────────────────────────
def extract_stim_segments(arr: np.ndarray, onsets: np.ndarray, fs: float):
    """Stim segments: arr[onset : onset + STIM_DUR_S*fs]."""
    T        = arr.shape[0]
    dur_samp = int(STIM_DUR_S * fs)
    return [arr[on:on + dur_samp] for on in onsets if on + dur_samp <= T]


def extract_sham_pre(arr: np.ndarray, onsets: np.ndarray, fs: float):
    """Sham def. 1: 5 s immediately preceding each stim onset."""
    win_samp = int(SHAM_WIN_S * fs)
    return [arr[on - win_samp:on] for on in onsets if on - win_samp >= 0]


def extract_sham_between(arr: np.ndarray, onsets: np.ndarray, offsets: np.ndarray, fs: float):
    """Sham def. 2: 5 s centered in the gap between consecutive stim blocks."""
    T        = arr.shape[0]
    win_samp = int(SHAM_WIN_S * fs)
    segs = []
    for i in range(len(onsets) - 1):
        gap_start = offsets[i]
        gap_end   = onsets[i + 1]
        gap_len   = gap_end - gap_start
        if gap_len < win_samp:
            continue
        mid   = gap_start + gap_len // 2
        start = mid - win_samp // 2
        end   = start + win_samp
        if start < 0 or end > T:
            continue
        segs.append(arr[start:end])
    return segs


def extract_sham_baseline_chop(arr: np.ndarray, first_onset: int, fs: float):
    """Sham def. 3: chop the leading baseline (recording start -> first stim
    onset) into consecutive, non-overlapping 5 s windows."""
    win_samp = int(SHAM_WIN_S * fs)
    n_wins   = int(first_onset // win_samp)
    return [arr[i * win_samp:(i + 1) * win_samp] for i in range(n_wins)]


# ── Feature extraction ────────────────────────────────────────────────────────
def mua_features(segs):
    """Mean firing rate per channel  →  (N, 32)"""
    return [s.mean(axis=0) for s in segs]


def lfp_features(segs):
    """RMS amplitude per channel  →  (N, 32)"""
    return [np.sqrt((s ** 2).mean(axis=0)) for s in segs]


# ── Classification ────────────────────────────────────────────────────────────
def build_classifier():
    return Pipeline([
        ('sc',  StandardScaler()),
        ('clf', SVC(kernel='linear', class_weight='balanced',
                    probability=True, random_state=42))])


def run_loo(X: np.ndarray, y: np.ndarray):
    """
    Trial-level Leave-One-Out CV: each fold leaves out exactly one trial
    (not one session) — sessions have already been pooled into X/y before
    this is called.
    Returns None if too few trials/classes, else dict with y_true, y_pred, acc (0-100).
    """
    if len(np.unique(y)) < 2 or len(y) < 4:
        return None

    loo    = LeaveOneOut()
    y_pred = np.zeros(len(y), dtype=int)
    for tr, te in loo.split(X):
        mdl = build_classifier()
        mdl.fit(X[tr], y[tr])
        y_pred[te] = mdl.predict(X[te])
    return dict(y_true=y, y_pred=y_pred, acc=(y_pred == y).mean() * 100)


# ── Plotting helpers ──────────────────────────────────────────────────────────
def _bar_with_sem_ttest(ax, arr_s, arr_h, ylabel, title):
    """Bar chart (mean ± SEM) of stim vs. sham per channel, with a per-channel
    Welch's t-test; significant channels are starred above their bars."""
    x = np.arange(N_CH)
    if arr_s is None or len(arr_s) == 0 or arr_h is None or len(arr_h) == 0:
        ax.text(0.5, 0.5, 'data not available for this subject',
                ha='center', va='center', transform=ax.transAxes, fontsize=11,
                color='gray')
        ax.set_title(title)
        ax.set_xticks(x); ax.set_xticklabels(x, fontsize=7)
        ax.set_xlabel('Channel')
        return

    arr_s, arr_h  = np.array(arr_s), np.array(arr_h)
    mean_s, mean_h = arr_s.mean(axis=0), arr_h.mean(axis=0)
    sem_s,  sem_h  = stats.sem(arr_s, axis=0), stats.sem(arr_h, axis=0)
    _, pvals       = stats.ttest_ind(arr_s, arr_h, axis=0, equal_var=False)

    ax.bar(x - 0.2, mean_s, 0.4, yerr=sem_s, capsize=2, label='Stim',
           color='steelblue', alpha=0.85, error_kw=dict(lw=0.8))
    ax.bar(x + 0.2, mean_h, 0.4, yerr=sem_h, capsize=2, label='Sham',
           color='salmon', alpha=0.85, error_kw=dict(lw=0.8))

    y_top = np.maximum(mean_s + sem_s, mean_h + sem_h)
    y_bot = np.minimum(mean_s - sem_s, mean_h - sem_h)
    span  = (y_top.max() - y_bot.min()) or 1.0
    for ch in range(N_CH):
        p = pvals[ch]
        if np.isnan(p):
            continue
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else None
        if star:
            ax.text(x[ch], y_top[ch] + 0.02 * span, star, ha='center',
                    va='bottom', fontsize=8)

    ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels(x, fontsize=7)
    ax.set_xlabel('Channel'); ax.set_ylabel(ylabel)
    ax.legend(fontsize=10)
    n_sig = int(np.sum(pvals[~np.isnan(pvals)] < 0.05))
    print(f'    {title}: {n_sig}/{N_CH} channels significant (p<0.05, Welch t-test)')


def plot_activity(sub_name, X_lfp_s, X_lfp_h, X_mua_s, X_mua_h):
    """Mean ± SEM activity per channel, stim vs. pre-stim sham, for one subject,
    with a per-channel Welch's t-test (stim vs. sham)."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle(f'{sub_name}  –  Mean Activity per Channel (Stim vs. Pre-Stim Sham)\n'
                 "(error bars = SEM;  * p<0.05  ** p<0.01  *** p<0.001, Welch's t-test)",
                 fontsize=13, fontweight='bold')

    _bar_with_sem_ttest(axes[0], X_mua_s, X_mua_h, 'Mean Firing Rate (a.u.)',
                        'MUA – Mean Firing Rate per Channel')
    _bar_with_sem_ttest(axes[1], X_lfp_s, X_lfp_h, 'RMS Amplitude (µV)',
                        'LFP – RMS Amplitude per Channel')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'{sub_name}_activity.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {out}')


def plot_confusion_grid(sub_name, results):
    """
    2 (rows: LFP, MUA) x 3 (cols: sham-block definition) grid of confusion
    matrices. Each cell is normalized per true label (row-normalized) and
    shown as a percentage: what proportion of stim trials were labeled
    correctly vs. incorrectly, and likewise for sham trials.
    results[modality][sham_key] = {'y_true','y_pred','acc'} or None.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'{sub_name}  –  LOO Decoding Confusion Matrices (Stim vs. Sham)\n'
                 '(rows: signal modality  ·  columns: sham-block definition  ·  '
                 'values: % of true-label trials)',
                 fontsize=12, fontweight='bold')

    for r, modality in enumerate(MODALITIES):
        for c, (sham_key, sham_label) in enumerate(SHAM_DEFINITIONS):
            ax  = axes[r, c]
            res = results.get(modality, {}).get(sham_key)

            if r == 0:
                ax.set_title(sham_label, fontsize=11, fontweight='bold')

            if res is None:
                ax.text(0.5, 0.5, 'insufficient\ndata', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
            else:
                cm_pct = confusion_matrix(res['y_true'], res['y_pred'],
                                           normalize='true') * 100
                ConfusionMatrixDisplay(cm_pct, display_labels=['Sham', 'Stim']).plot(
                    ax=ax, colorbar=False, cmap='Blues', values_format='.1f')
                for txt in ax.texts:
                    txt.set_text(txt.get_text() + '%')
                ax.set_xlabel(f'Predicted label\nLOO Acc = {res["acc"]:.1f}%', fontsize=9)

            ax.set_ylabel(f'{modality}\nTrue label' if c == 0 else '',
                          fontsize=10 if c == 0 else 9,
                          fontweight='bold' if c == 0 else 'normal')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'{sub_name}_confusion_grid.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {out}')


def plot_summary_grid(df, subjects):
    """Final 2x3 grid (same layout as plot_confusion_grid) of per-subject
    LOO accuracy bar charts, one panel per (modality, sham definition)."""
    sub_labels = [os.path.basename(s) for s in subjects]
    x_pos      = np.arange(len(sub_labels))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    fig.suptitle('LOO Decoding Accuracy: Stim vs. Sham (per subject)\n'
                 '(rows: signal modality  ·  columns: sham-block definition)',
                 fontsize=13, fontweight='bold')

    for r, modality in enumerate(MODALITIES):
        for c, (sham_key, sham_label) in enumerate(SHAM_DEFINITIONS):
            ax     = axes[r, c]
            sub_df = df[(df['modality'] == modality) & (df['sham'] == sham_key)]

            vals = []
            for sub in sub_labels:
                row = sub_df[sub_df['subject'] == sub]
                vals.append(float(row['acc'].values[0]) if len(row) else np.nan)

            color = 'steelblue' if modality == 'LFP' else 'salmon'
            ax.bar(x_pos, vals, 0.6, color=color, alpha=0.85)
            ax.axhline(50, color='red', lw=1, ls='--')
            ax.set_ylim(0, 105)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([s.split('-')[-1] for s in sub_labels],
                               rotation=30, ha='right', fontsize=8)
            if r == 0:
                ax.set_title(sham_label, fontsize=11, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{modality}\nLOO Accuracy (%)', fontsize=10, fontweight='bold')
            if r == 1:
                ax.set_xlabel('Subject')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'summary_accuracy_grid.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n[saved] {out}')


# ── Main per-subject analysis loop ────────────────────────────────────────────
subjects    = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*')))
summary_acc = []          # (sub_name, modality, sham_key, acc)

for sub_path in subjects:
    sub_name = os.path.basename(sub_path)
    print(f'\n{"="*65}')
    print(f'  SUBJECT: {sub_name}')
    print(f'{"="*65}')

    # Discover stim numbers (sessions) from noArti files (ground truth channel)
    hf_files  = glob.glob(os.path.join(sub_path, 'stim*_noArti.mat'))
    stim_nums = sorted({
        int(re.search(r'stim(\d+)_', os.path.basename(f)).group(1))
        for f in hf_files
    })
    print(f'  Sessions found: {stim_nums}')

    # Pool trial features across all sessions for this subject.
    # feats[modality]['stim' | 'pre' | 'between' | 'baseline'] -> list of (32,) feature vecs
    feats = {mod: {'stim': [], 'pre': [], 'between': [], 'baseline': []}
             for mod in MODALITIES}
    has_mua = False

    for sn in stim_nums:
        dat = load_stim_file(sub_path, sn)
        if dat is None:
            print(f'  stim{sn}: file missing, skipping')
            continue

        ons, offs = detect_stim_blocks(dat['stim'], dat['fs'])
        fs = dat['fs']
        print(f'  stim{sn}: {len(ons)} stim blocks detected  '
              f'(MUA={"yes" if dat["mua"] is not None else "no"})')
        if len(ons) == 0:
            continue

        # LFP (always available)
        feats['LFP']['stim'].extend(lfp_features(extract_stim_segments(dat['lfp'], ons, fs)))
        feats['LFP']['pre'].extend(lfp_features(extract_sham_pre(dat['lfp'], ons, fs)))
        feats['LFP']['between'].extend(lfp_features(extract_sham_between(dat['lfp'], ons, offs, fs)))
        feats['LFP']['baseline'].extend(lfp_features(extract_sham_baseline_chop(dat['lfp'], ons[0], fs)))

        # MUA (if available for this session)
        if dat['mua'] is not None:
            has_mua = True
            feats['MUA']['stim'].extend(mua_features(extract_stim_segments(dat['mua'], ons, fs)))
            feats['MUA']['pre'].extend(mua_features(extract_sham_pre(dat['mua'], ons, fs)))
            feats['MUA']['between'].extend(mua_features(extract_sham_between(dat['mua'], ons, offs, fs)))
            feats['MUA']['baseline'].extend(mua_features(extract_sham_baseline_chop(dat['mua'], ons[0], fs)))

    n_stim_lfp = len(feats['LFP']['stim'])
    print(f'\n  LFP stim trials pooled: {n_stim_lfp}')
    if n_stim_lfp < 2:
        print('  Insufficient data — skipping subject')
        continue

    # ── Activity plot (stim vs. pre-stim sham; mean ± SEM, per-channel t-test) ──
    plot_activity(
        sub_name,
        feats['LFP']['stim'], feats['LFP']['pre'],
        feats['MUA']['stim'] if has_mua else None,
        feats['MUA']['pre']  if has_mua else None,
    )

    if not RUN_DECODE:
        continue

    # ── Decode: per modality, per sham definition, trial-level LOO ──────
    results = {mod: {} for mod in MODALITIES}
    for modality in MODALITIES:
        if modality == 'MUA' and not has_mua:
            for sham_key, _ in SHAM_DEFINITIONS:
                results[modality][sham_key] = None
            continue

        X_stim = np.array(feats[modality]['stim'])
        for sham_key, sham_label in SHAM_DEFINITIONS:
            X_sham = np.array(feats[modality][sham_key])
            n = min(len(X_stim), len(X_sham))
            if n < 2:
                print(f'  [{modality} | {sham_label}] insufficient sham trials ({len(X_sham)}) — skipping')
                results[modality][sham_key] = None
                continue

            X = np.vstack([X_stim[:n], X_sham[:n]])
            y = np.array([1] * n + [0] * n)
            res = run_loo(X, y)
            results[modality][sham_key] = res
            if res is None:
                print(f'  [{modality} | {sham_label}] n={n}/class — too few trials for LOO')
                continue

            print(f'  [{modality} | {sham_label}] n={n}/class  LOO acc = {res["acc"]:.1f}%')
            print(classification_report(res['y_true'], res['y_pred'],
                                        target_names=['Sham', 'Stim'], zero_division=0))
            summary_acc.append((sub_name, modality, sham_key, res['acc']))

    # ── Confusion matrix grid ────────────────────────────────────────────
    plot_confusion_grid(sub_name, results)


# ── Summary across subjects ────────────────────────────────────────────────────
if summary_acc:
    df = pd.DataFrame(summary_acc, columns=['subject', 'modality', 'sham', 'acc'])
    plot_summary_grid(df, subjects)

    print('\n=== FINAL SUMMARY ===')
    sham_label_map = dict(SHAM_DEFINITIONS)
    df['sham_label'] = df['sham'].map(sham_label_map)
    print(df.pivot_table(index=['subject', 'modality', 'sham_label'],
                         values='acc').to_string() + '%')

    df.drop(columns='sham_label').to_csv(
        os.path.join(OUTPUT_DIR, 'decode_stim_sham_results.csv'), index=False)
    print(f"\n[saved] {os.path.join(OUTPUT_DIR, 'decode_stim_sham_results.csv')}")
elif not RUN_DECODE:
    print('\nRUN_DECODE=False — skipped LOO decoding / confusion-matrix summary.')
else:
    print('\nNo results to summarize.')

print('\nDone. Output saved to:', OUTPUT_DIR)
