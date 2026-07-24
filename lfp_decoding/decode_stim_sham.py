#!/usr/bin/env python3
"""
decode_stim_sham.py

Decode stimulation vs. sham using multiunit spiking activity (MUA) and LFP.
  - 5 subjects from /Volumes/SanDisk/files/derivative/
  - Stim blocks detected with adaptive RMS thresholding
  - Feature extraction: mean firing rate (MUA) + RMS amplitude (LFP) per channel
  - Classifiers: Random Forest, LDA, SVM  ·  Leave-One-Out CV
  - Per-subject: activity bar chart (stim vs sham) + confusion matrices
  - Summary: accuracy bar chart across subjects and modalities
"""

import os, glob, re, warnings
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
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
PRE_STIM_S   = 5.0     # seconds before stim onset  (sham window)
RMS_WIN_S    = 0.5     # RMS smoothing window
N_CH         = 32

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
    Returns (onsets_samp, offsets_samp).
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
def extract_windows(arr: np.ndarray, onsets: np.ndarray, fs: float):
    """
    For each onset, return:
      stim seg  : arr[onset : onset + STIM_DUR_S*fs]
      sham seg  : arr[onset - PRE_STIM_S*fs : onset]
    Skips blocks with out-of-bounds indices.
    """
    T        = arr.shape[0]
    dur_samp = int(STIM_DUR_S  * fs)
    pre_samp = int(PRE_STIM_S  * fs)
    stim_s, sham_s = [], []
    for on in onsets:
        if on + dur_samp > T or on - pre_samp < 0:
            continue
        stim_s.append(arr[on           : on + dur_samp])
        sham_s.append(arr[on - pre_samp : on          ])
    return stim_s, sham_s


# ── Feature extraction ────────────────────────────────────────────────────────
def mua_features(segs):
    """Mean firing rate per channel  →  (N, 32)"""
    return np.array([s.mean(axis=0) for s in segs])


def lfp_features(segs):
    """RMS amplitude per channel  →  (N, 32)"""
    return np.array([np.sqrt((s ** 2).mean(axis=0)) for s in segs])


# ── Classification ────────────────────────────────────────────────────────────
CLF_NAMES = ['Random Forest', 'LDA', 'SVM']


def build_classifiers():
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced',
            max_features='sqrt', random_state=42),
        'LDA': Pipeline([
            ('sc',  StandardScaler()),
            ('clf', LinearDiscriminantAnalysis())]),
        'SVM': Pipeline([
            ('sc',  StandardScaler()),
            ('clf', SVC(kernel='rbf', class_weight='balanced',
                        probability=True, random_state=42))]),
    }


def run_loo(X: np.ndarray, y: np.ndarray):
    """
    Leave-One-Out CV for all three classifiers.
    Returns dict  {clf_name: {'acc': float, 'y_pred': ndarray}}
    or empty dict if too few samples.
    """
    if len(np.unique(y)) < 2 or len(y) < 4:
        return {}

    loo     = LeaveOneOut()
    models  = build_classifiers()
    results = {}
    for name, mdl in models.items():
        y_pred = np.zeros(len(y), dtype=int)
        for tr, te in loo.split(X):
            mdl.fit(X[tr], y[tr])
            y_pred[te] = mdl.predict(X[te])
        results[name] = dict(acc=(y_pred == y).mean(), y_pred=y_pred)
    return results


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_activity(sub_name, X_lfp_s, X_lfp_h, X_mua_s, X_mua_h):
    """Bar chart of mean activity per channel, stim vs sham, for one subject."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(f'{sub_name}  –  Mean Activity per Channel (Stim vs. Sham)',
                 fontsize=13, fontweight='bold')
    x = np.arange(N_CH)

    # MUA panel
    ax = axes[0]
    if X_mua_s is not None and len(X_mua_s) > 0:
        mu_s = np.array(X_mua_s).mean(axis=0)
        mu_h = np.array(X_mua_h).mean(axis=0)
        ax.bar(x - 0.2, mu_s, 0.4, label='Stim', color='steelblue', alpha=0.85)
        ax.bar(x + 0.2, mu_h, 0.4, label='Sham', color='salmon',    alpha=0.85)
        ax.set_ylabel('Mean Firing Rate (a.u.)')
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'MUA data not available for this subject',
                ha='center', va='center', transform=ax.transAxes, fontsize=11,
                color='gray')
    ax.set_title('MUA – Mean Firing Rate per Channel')
    ax.set_xticks(x); ax.set_xticklabels(x, fontsize=7)
    ax.set_xlabel('Channel')

    # LFP panel
    ax = axes[1]
    rms_s = np.array(X_lfp_s).mean(axis=0)
    rms_h = np.array(X_lfp_h).mean(axis=0)
    ax.bar(x - 0.2, rms_s, 0.4, label='Stim', color='steelblue', alpha=0.85)
    ax.bar(x + 0.2, rms_h, 0.4, label='Sham', color='salmon',    alpha=0.85)
    ax.set_xlabel('Channel'); ax.set_ylabel('RMS Amplitude (µV)')
    ax.set_title('LFP – RMS Amplitude per Channel')
    ax.set_xticks(x); ax.set_xticklabels(x, fontsize=7)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'{sub_name}_activity.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {out}')


def plot_confusion(sub_name, modal_results):
    """
    Grid of confusion matrices: rows = modalities, cols = classifiers.
    modal_results: list of (modal_name, y_true, clf_result_dict)
    """
    valid = [(m, y, r) for m, y, r in modal_results if r]
    if not valid:
        return

    nrows = len(valid)
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 4.5 * nrows), squeeze=False)
    fig.suptitle(f'{sub_name}  –  LOO Decoding Confusion Matrices', fontsize=13,
                 fontweight='bold')

    for r, (mname, y_true, res) in enumerate(valid):
        for c, cname in enumerate(CLF_NAMES):
            ax = axes[r, c]
            if cname in res:
                cm = confusion_matrix(y_true, res[cname]['y_pred'])
                ConfusionMatrixDisplay(cm, display_labels=['Sham', 'Stim']).plot(
                    ax=ax, colorbar=False, cmap='Blues')
                ax.set_title(
                    f'{mname}  |  {cname}\nLOO Acc = {res[cname]["acc"]:.2f}',
                    fontsize=10)
            else:
                ax.set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'{sub_name}_confusion.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [saved] {out}')


# ── Main per-subject analysis loop ────────────────────────────────────────────
subjects    = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*')))
summary_acc = []          # (sub_name, modality, clf_name, acc)

for sub_path in subjects:
    sub_name = os.path.basename(sub_path)
    print(f'\n{"="*65}')
    print(f'  SUBJECT: {sub_name}')
    print(f'{"="*65}')

    # Discover stim numbers from noArti files (ground truth channel)
    hf_files  = glob.glob(os.path.join(sub_path, 'stim*_noArti.mat'))
    stim_nums = sorted({
        int(re.search(r'stim(\d+)_', os.path.basename(f)).group(1))
        for f in hf_files
    })
    print(f'  Stim conditions found: {stim_nums}')

    # Accumulate segments across all stim files
    lfp_feats_stim, lfp_feats_sham = [], []
    mua_feats_stim, mua_feats_sham = [], []
    has_mua = False

    for sn in stim_nums:
        dat = load_stim_file(sub_path, sn)
        if dat is None:
            print(f'  stim{sn}: file missing, skipping')
            continue

        ons, offs = detect_stim_blocks(dat['stim'], dat['fs'])
        print(f'  stim{sn}: {len(ons)} stim blocks detected  '
              f'(MUA={"yes" if dat["mua"] is not None else "no"})')

        # LFP
        ls, lh = extract_windows(dat['lfp'], ons, dat['fs'])
        if ls:
            lfp_feats_stim.extend(lfp_features(ls))
            lfp_feats_sham.extend(lfp_features(lh))

        # MUA
        if dat['mua'] is not None:
            ms, mh = extract_windows(dat['mua'], ons, dat['fs'])
            if ms:
                mua_feats_stim.extend(mua_features(ms))
                mua_feats_sham.extend(mua_features(mh))
                has_mua = True

    nL = min(len(lfp_feats_stim), len(lfp_feats_sham))
    nM = min(len(mua_feats_stim), len(mua_feats_sham))
    print(f'\n  LFP windows: {nL} stim + {nL} sham')
    print(f'  MUA windows: {nM} stim + {nM} sham')

    if nL < 2:
        print('  Insufficient data — skipping subject')
        continue

    # Build numpy arrays
    Xl = np.vstack([np.array(lfp_feats_stim[:nL]), np.array(lfp_feats_sham[:nL])])
    yl = np.array([1] * nL + [0] * nL)

    Xm = ym = Xc = yc = None
    if nM >= 4:
        Xm = np.vstack([np.array(mua_feats_stim[:nM]), np.array(mua_feats_sham[:nM])])
        ym = np.array([1] * nM + [0] * nM)
        nc = min(nL, nM)
        Xc = np.hstack([
            np.vstack([np.array(lfp_feats_stim[:nc]), np.array(lfp_feats_sham[:nc])]),
            np.vstack([np.array(mua_feats_stim[:nc]), np.array(mua_feats_sham[:nc])])
        ])
        yc = np.array([1] * nc + [0] * nc)

    # ── Activity plot ──────────────────────────────────────────────────
    plot_activity(
        sub_name,
        lfp_feats_stim[:nL], lfp_feats_sham[:nL],
        mua_feats_stim[:nM] if nM >= 1 else None,
        mua_feats_sham[:nM] if nM >= 1 else None,
    )

    # ── Decode ────────────────────────────────────────────────────────
    modal_data   = [('LFP', Xl, yl), ('MUA', Xm, ym), ('MUA+LFP', Xc, yc)]
    modal_labels = ['LFP', 'MUA', 'MUA+LFP']
    all_res      = {}

    for mname, X, y in modal_data:
        if X is None:
            all_res[mname] = {}
            continue
        print(f'\n  [{mname}]  n={len(y)//2} per class, {X.shape[1]} features')
        res = run_loo(X, y)
        all_res[mname] = res
        for cname, r in res.items():
            print(f'    {cname}: acc = {r["acc"]:.2f}')
            print(classification_report(y, r['y_pred'],
                                        target_names=['Sham', 'Stim'],
                                        zero_division=0))
            summary_acc.append((sub_name, mname, cname, r['acc']))

    # ── Confusion matrix plot ──────────────────────────────────────────
    plot_confusion(sub_name, [
        ('LFP',     yl, all_res.get('LFP',     {})),
        ('MUA',     ym, all_res.get('MUA',     {})),
        ('MUA+LFP', yc, all_res.get('MUA+LFP', {})),
    ])


# ── Summary plot ──────────────────────────────────────────────────────────────
if summary_acc:
    import pandas as pd
    df = pd.DataFrame(summary_acc, columns=['subject', 'modality', 'classifier', 'acc'])

    sub_labels = [os.path.basename(s) for s in subjects]
    modalities = ['LFP', 'MUA', 'MUA+LFP']
    colors     = {'Random Forest': '#4878CF', 'LDA': '#6ACC65', 'SVM': '#D65F5F'}

    fig, axes = plt.subplots(1, len(modalities), figsize=(18, 6), sharey=True)
    fig.suptitle('LOO Decoding Accuracy: Stim vs. Sham  (per subject, per modality)',
                 fontsize=13, fontweight='bold')

    for ax, mod in zip(axes, modalities):
        sub_df = df[df['modality'] == mod]
        x_pos  = np.arange(len(sub_labels))
        width  = 0.25

        for i, cname in enumerate(CLF_NAMES):
            c_df = sub_df[sub_df['classifier'] == cname]
            vals = []
            for sub in sub_labels:
                row = c_df[c_df['subject'] == sub]
                vals.append(float(row['acc'].values[0]) if len(row) else np.nan)
            ax.bar(x_pos + i * width, vals, width, label=cname,
                   color=colors[cname], alpha=0.85)

        ax.axhline(0.5, color='red', lw=1, ls='--', label='Chance')
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels([s.split('-')[-1] for s in sub_labels],
                           rotation=30, ha='right', fontsize=8)
        ax.set_title(mod, fontsize=11, fontweight='bold')
        ax.set_ylabel('LOO Accuracy' if mod == 'LFP' else '')
        ax.legend(fontsize=8)
        ax.set_xlabel('Subject')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'summary_accuracy.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n[saved] {out}')
    print('\n=== FINAL SUMMARY ===')
    print(df.pivot_table(index=['subject', 'modality'], columns='classifier',
                         values='acc').to_string())
else:
    print('\nNo results to summarize.')

print('\nDone. Output saved to:', OUTPUT_DIR)
