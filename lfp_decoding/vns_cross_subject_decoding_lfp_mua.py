#!/usr/bin/env python3
"""
vns_cross_subject_decoding_lfp_mua.py

Cross-subject VNS decoding from LFP and MUA channel activity (32 channels per
subject), replicating the feature-processing / decoding pipeline of
single_unit_decoding/vns_cross_subject_decoding.ipynb but using continuous
LFP / MUA signals in place of Kilosort single-unit spike trains.

For every `stim{N}_noArti.mat` file under DATA_ROOT:
  1. Recover each VNS train's onset time, amplitude, and pulse width directly
     from the `stim_DAQ` trace (extract_train_params — unchanged from the
     notebook).
  2. Anchor a matched Sham window at the midpoint of the gap since the
     *previous* train (same adaptive-baseline logic as the notebook).
  3. Build 500 ms binned feature vectors per channel:
       - LFP → RMS amplitude per bin per channel
       - MUA → mean activity per bin per channel
     (channel-major flatten, mirroring the notebook's per-unit binned
     spike-count vectors.)
  4. Decode each contrast with enough trials via linear SVM + LOO-CV +
     permutation test (svm_decode — unchanged from the notebook):
       - VNS vs Sham
       - Amplitude level (k-means clustered, ≥2 levels with ≥3 trials each)
       - Pulse width (k-means clustered, ≥2 levels with ≥3 trials each)
  5. Aggregate across sessions/subjects, separately for LFP and MUA, and plot
     accuracy vs. permutation null (box plots), a paired-t-test summary
     table, and a per-session accuracy pivot table.
"""

import os, glob, re, warnings
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, permutation_test_score
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT   = '/Volumes/SanDisk/files/derivative'
OUTPUT_DIR  = os.path.expanduser('~/Desktop/NTS_decoding_results')
N_CH        = 32
WIN_DUR     = 5.0    # s, duration of a VNS train (and of the matched Sham window)
BIN_SIZE    = 0.5    # s, width of each bin used as a decoding feature
N_BINS      = int(WIN_DUR / BIN_SIZE)
MIN_TRIALS  = 3      # minimum trials required per class before a contrast is decoded
N_PERMS     = 500     # permutations per condition for the null (chance) distribution

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_stim_file(sub_dir: str, stim_num: int):
    """
    Load LFP (noArti), MUA (MUA_100 > mua_50 fallback), and the raw stim_DAQ
    trace for one stim file.
    Returns dict with keys: lfp, mua, stim_daq, time, fs  –– or None if missing.
    """
    base     = os.path.join(sub_dir, f'stim{stim_num}')
    lfp_path = base + '_noArti.mat'
    if not os.path.exists(lfp_path):
        return None

    with h5py.File(lfp_path, 'r') as f:
        lfp  = np.array(f['data']).squeeze()    # (T, 32)
        time = np.array(f['time']).squeeze()    # (T,)
        fs   = float(np.array(f['fs']).flat[0])

        if 'stim_DAQ' in f:
            stim_daq = np.array(f['stim_DAQ']['data']).squeeze()
        else:
            stim_daq = np.array(f['stim']).squeeze()

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
        time     = time[:n]
        stim_daq = stim_daq[:n] if stim_daq.shape[0] >= n else stim_daq
        mua      = mua[:n]

    return dict(lfp=lfp, mua=mua, stim_daq=stim_daq, time=time, fs=fs)


# ── Train-parameter extraction (unchanged from vns_cross_subject_decoding.ipynb) ──
def extract_train_params(stim_daq, time_vec):
    """Detect VNS pulse trains in the raw stim_DAQ trace and return, per train, its onset
    time plus amplitude / pulse-width (used as decoding targets below)."""
    abs_sig = np.abs(stim_daq)
    abs_max = abs_sig.max()
    if abs_max == 0:
        return None
    thresh    = abs_max * 0.3
    binary    = (abs_sig > thresh).astype(int)
    pulse_on  = np.where(np.diff(binary) == 1)[0] + 1
    pulse_off = np.where(np.diff(binary) == -1)[0] + 1
    if len(pulse_on) < 5:
        return None
    m = min(len(pulse_on), len(pulse_off))
    pulse_on, pulse_off = pulse_on[:m], pulse_off[:m]
    pulse_t = time_vec[pulse_on]
    gaps    = np.diff(pulse_t)
    ts_idx  = np.concatenate([[0], np.where(gaps > 0.5)[0] + 1])
    te_idx  = np.concatenate([np.where(gaps > 0.5)[0], [m - 1]])
    onset_times, raw_amps, pw_ms = [], [], []
    for ts, te in zip(ts_idx, te_idx):
        p_on  = pulse_on[ts:te + 1]
        p_off = pulse_off[ts:te + 1]
        amp = np.mean([abs_sig[pi:po + 1].max() for pi, po in zip(p_on, p_off)])
        raw_amps.append(amp)
        pw = np.median(time_vec[p_off] - time_vec[p_on]) * 1000
        pw_ms.append(pw)
        onset_times.append(pulse_t[ts])
    raw_amps = np.array(raw_amps)
    return dict(onset_times=np.array(onset_times),
                norm_amps=raw_amps / raw_amps.max(),
                raw_amps=raw_amps,
                pw_ms=np.array(pw_ms))


def cluster_to_labels(values, tol=0.15):
    """Group a 1-D continuous parameter (amplitude, pulse width, ...) into a small number of
    discrete k-means levels, picking the smallest k whose within-cluster spread (relative to
    the value range) is below tol."""
    v   = values.reshape(-1, 1)
    rng = values.max() - values.min()
    if rng < 1e-6:
        return np.zeros(len(values), dtype=int), np.array([values.mean()]), 1
    for k in range(1, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(v)
        if km.inertia_ ** 0.5 / rng < tol:
            break
    km        = KMeans(n_clusters=k, n_init=10, random_state=0).fit(v)
    order     = np.argsort(km.cluster_centers_.flatten())
    label_map = {old: new for new, old in enumerate(order)}
    labels    = np.array([label_map[l] for l in km.labels_])
    centers   = np.sort(km.cluster_centers_.flatten())
    return labels, centers, k


# ── Feature extraction ────────────────────────────────────────────────────────
def binned_channel_features(window: np.ndarray, fs: float, stat: str):
    """
    Bin a (T, n_ch) window into N_BINS bins of BIN_SIZE seconds and compute a
    per-bin, per-channel statistic, flattened channel-major:
    [ch0_bin0..ch0_binN, ch1_bin0..ch1_binN, ...]  (mirrors binned_spike_counts).
      stat='rms'  → RMS amplitude per bin   (LFP)
      stat='mean' → mean activity per bin   (MUA)
    """
    bin_samp = int(BIN_SIZE * fs)
    n_ch     = window.shape[1]
    feats    = []
    for ch in range(n_ch):
        sig = window[:N_BINS * bin_samp, ch].reshape(N_BINS, bin_samp)
        if stat == 'rms':
            vals = np.sqrt((sig ** 2).mean(axis=1))
        else:
            vals = sig.mean(axis=1)
        feats.append(vals)
    return np.concatenate(feats)


# ── Decoding (unchanged from vns_cross_subject_decoding.ipynb) ────────────────
def svm_decode(X, y):
    """
    Linear SVM with LOO-CV + permutation test.
    Returns (accuracy, perm_mean, perm_std, p_value) or all-nan if too few trials.
    """
    if len(X) < 2 * MIN_TRIALS or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, np.nan
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))
    loo = LeaveOneOut()
    score, perm_scores, pvalue = permutation_test_score(
        clf, X, y, cv=loo, n_permutations=N_PERMS,
        scoring='accuracy', random_state=42, n_jobs=1
    )
    return score, perm_scores.mean(), perm_scores.std(), pvalue


# ── Main loop: build binned features + decode, per session, per modality ──────
records  = []
subjects = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*')))

for sub_path in subjects:
    sub_name = os.path.basename(sub_path)

    hf_files  = glob.glob(os.path.join(sub_path, 'stim*_noArti.mat'))
    stim_nums = sorted({
        int(re.search(r'stim(\d+)_', os.path.basename(f)).group(1))
        for f in hf_files
    })

    for sn in stim_nums:
        stim_name = f'stim{sn}'
        dat = load_stim_file(sub_path, sn)
        if dat is None:
            continue

        fs, time_vec = dat['fs'], dat['time']
        params = extract_train_params(dat['stim_daq'], time_vec)
        if params is None or len(params['onset_times']) < MIN_TRIALS:
            continue

        train_onsets = params['onset_times']
        T            = dat['lfp'].shape[0]
        win_samp     = int(WIN_DUR * fs)
        has_mua      = dat['mua'] is not None

        X_vns_lfp, X_base_lfp = [], []
        X_vns_mua, X_base_mua = [], []
        valid_onsets = []

        for i, onset in enumerate(train_onsets):
            if i == 0:
                continue   # no preceding train to anchor a matched baseline midpoint
            gap = onset - train_onsets[i - 1]
            if gap / 2 < WIN_DUR:
                continue   # trains too close together to fit a clean, non-overlapping baseline window
            base_start = train_onsets[i - 1] + gap / 2 - WIN_DUR / 2
            if base_start < 0:
                continue

            onset_idx = np.searchsorted(time_vec, onset)
            base_idx  = np.searchsorted(time_vec, base_start)
            if onset_idx + win_samp > T or base_idx + win_samp > T or base_idx < 0:
                continue

            X_vns_lfp.append(binned_channel_features(
                dat['lfp'][onset_idx:onset_idx + win_samp], fs, 'rms'))
            X_base_lfp.append(binned_channel_features(
                dat['lfp'][base_idx:base_idx + win_samp], fs, 'rms'))

            if has_mua:
                X_vns_mua.append(binned_channel_features(
                    dat['mua'][onset_idx:onset_idx + win_samp], fs, 'mean'))
                X_base_mua.append(binned_channel_features(
                    dat['mua'][base_idx:base_idx + win_samp], fs, 'mean'))

            valid_onsets.append(onset)

        if len(X_vns_lfp) < MIN_TRIALS:
            continue

        valid_onsets = np.array(valid_onsets)
        valid_mask   = np.isin(train_onsets, valid_onsets)
        norm_amps    = params['norm_amps'][valid_mask]
        pw_ms        = params['pw_ms'][valid_mask]

        base_rec = dict(subject=sub_name, stim=stim_name,
                         n_channels=N_CH, n_vns_trains=len(X_vns_lfp))

        def add(modality, decode, X, y, chance_k):
            acc, pm, ps, pv = svm_decode(X, y)
            records.append({**base_rec, 'modality': modality, 'decode': decode,
                             'accuracy': acc, 'perm_mean': pm, 'perm_std': ps,
                             'pvalue': pv, 'chance': 1 / chance_k, 'n_classes': chance_k})

        def decode_modality(modality, X_vns, X_base):
            X_vns  = np.array(X_vns)
            X_base = np.array(X_base)
            X_vb   = np.vstack([X_vns, X_base])
            y_vb   = np.array([1] * len(X_vns) + [0] * len(X_base))
            add(modality, 'VNS vs Sham', X_vb, y_vb, 2)

            amp_labels, _, n_amp = cluster_to_labels(norm_amps)
            if n_amp >= 2:
                counts = np.bincount(amp_labels)
                keep   = np.where(counts >= MIN_TRIALS)[0]
                if len(keep) >= 2:
                    mask  = np.isin(amp_labels, keep)
                    remap = {old: new for new, old in enumerate(keep)}
                    y_amp = np.array([remap[l] for l in amp_labels[mask]])
                    add(modality, 'Amplitude', X_vns[mask], y_amp, len(keep))

            pw_labels, _, n_pw = cluster_to_labels(pw_ms)
            if n_pw >= 2:
                counts = np.bincount(pw_labels)
                keep   = np.where(counts >= MIN_TRIALS)[0]
                if len(keep) >= 2:
                    mask  = np.isin(pw_labels, keep)
                    remap = {old: new for new, old in enumerate(keep)}
                    y_pw  = np.array([remap[l] for l in pw_labels[mask]])
                    add(modality, 'Pulse Width', X_vns[mask], y_pw, len(keep))
            return n_amp, n_pw

        n_amp, n_pw = decode_modality('LFP', X_vns_lfp, X_base_lfp)
        if has_mua and len(X_vns_mua) >= MIN_TRIALS:
            decode_modality('MUA', X_vns_mua, X_base_mua)

        vns_lfp_acc = next((r['accuracy'] for r in reversed(records)
                            if r['decode'] == 'VNS vs Sham' and r['modality'] == 'LFP'
                            and r['subject'] == sub_name and r['stim'] == stim_name), np.nan)
        print(f"{sub_name}/{stim_name}: trains={len(X_vns_lfp)}, "
              f"amp_lvls={n_amp}, pw_lvls={n_pw}, MUA={'yes' if has_mua else 'no'}, "
              f"LFP VNS={vns_lfp_acc:.0%}")

results = pd.DataFrame(records)
print(f"\nTotal sessions: {results[['subject','stim']].drop_duplicates().shape[0]}")
print(f"Total records:  {len(results)}")


# ── Box plot summary with significance (one figure per modality) ──────────────
def plot_summary(results, modality, out_path):
    decode_types = ['VNS vs Sham', 'Pulse Width', 'Amplitude']
    type_colors  = {'VNS vs Sham': '#2166ac', 'Pulse Width': '#4dac26', 'Amplitude': '#d6604d'}

    mod_df     = results[results['modality'] == modality]
    subjects   = sorted(mod_df['subject'].unique())
    sub_labels = [s.replace('sub-', '') for s in subjects]
    if not subjects:
        return

    fig, axes = plt.subplots(1, len(decode_types), figsize=(16, 6), sharey=True)
    fig.suptitle(
        f'{modality} decoding accuracy vs permutation null\n'
        '(filled dot = session p<0.05; bracket = paired t-test real vs shuffled)',
        fontsize=11, fontweight='bold'
    )

    for ax, dtype in zip(axes, decode_types):
        col = type_colors[dtype]
        df  = mod_df[mod_df['decode'] == dtype].dropna(subset=['accuracy', 'perm_mean']).copy()

        ax.set_title(dtype, fontweight='bold', color=col, fontsize=12)
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

        pos_real, pos_shuf, dat_real, dat_shuf, xticks, brackets = [], [], [], [], [], []

        for i_sub, sub in enumerate(subjects):
            sub_df = df[df['subject'] == sub]
            if sub_df.empty:
                continue
            real = sub_df['accuracy'].values
            shuf = sub_df['perm_mean'].values
            x_r  = i_sub * 2.2
            x_s  = x_r + 0.8
            pos_real.append(x_r); pos_shuf.append(x_s)
            dat_real.append(real); dat_shuf.append(shuf)
            xticks.append((x_r + x_s) / 2)

            if len(real) >= 3:
                _, pv = stats.ttest_rel(real, shuf, alternative='greater')
            elif len(real) == 2:
                pv = 0.25 if (real > shuf).all() else 1.0
            else:
                pv = 1.0
            star = ('***' if pv < 0.001 else '**' if pv < 0.01
                    else '*' if pv < 0.05 else 'ns')
            y_top = max(real.max(), shuf.max()) + 0.05
            brackets.append((x_r + 0.1, x_s - 0.1, y_top, star))

        if not dat_real:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            continue

        ax.boxplot(dat_real, positions=pos_real, widths=0.6, patch_artist=True,
                   showfliers=False,
                   medianprops=dict(color='white', linewidth=2),
                   boxprops=dict(facecolor=col, alpha=0.7),
                   whiskerprops=dict(color=col), capprops=dict(color=col))
        ax.boxplot(dat_shuf, positions=pos_shuf, widths=0.6, patch_artist=True,
                   showfliers=False,
                   medianprops=dict(color='k', linewidth=1.5),
                   boxprops=dict(facecolor='lightgray', alpha=0.85),
                   whiskerprops=dict(color='gray'), capprops=dict(color='gray'))

        for x_r, x_s, rv, sv, (_, sub) in zip(
                pos_real, pos_shuf, dat_real, dat_shuf,
                [(i, s) for i, s in enumerate(subjects)
                 if not df[df['subject'] == s].empty]):
            sub_df  = df[df['subject'] == sub]
            pvals_s = sub_df['pvalue'].values
            jit = np.random.uniform(-0.12, 0.12, len(rv))
            for j, (r, s, pv) in enumerate(zip(rv, sv, pvals_s)):
                ec = 'k' if (not np.isnan(pv) and pv < 0.05) else 'lightgray'
                fc = col if (not np.isnan(pv) and pv < 0.05) else 'white'
                ax.scatter(x_r + jit[j], r, s=50, color=fc,
                           edgecolors=ec, linewidths=1.3, zorder=5)
                ax.scatter(x_s + jit[j], s, s=30, color='gray',
                           edgecolors='dimgray', linewidths=0.8, zorder=5)
                ax.plot([x_r + jit[j], x_s + jit[j]], [r, s],
                        color='gray', lw=0.5, alpha=0.4, zorder=3)

        for xl, xr, yt, star in brackets:
            if star == 'ns':
                ax.text((xl + xr) / 2, yt + 0.01, 'ns',
                        ha='center', va='bottom', fontsize=8, color='gray')
            else:
                ax.plot([xl, xl, xr, xr], [yt, yt + 0.01, yt + 0.01, yt],
                        color='k', lw=1)
                ax.text((xl + xr) / 2, yt + 0.012, star,
                        ha='center', va='bottom', fontsize=10)

        ax.set_xticks(xticks)
        ax.set_xticklabels(sub_labels[:len(xticks)], fontsize=9)
        ax.set_xlabel('Subject')
        ax.set_xlim(-0.8, max(pos_shuf) + 0.8)

    axes[0].set_ylabel('LOO Accuracy')

    leg = [
        Patch(facecolor='steelblue', alpha=0.7,  label='Real accuracy'),
        Patch(facecolor='lightgray', alpha=0.85, label='Shuffled null'),
        Line2D([0], [0], marker='o', lw=0, markerfacecolor='steelblue',
               markeredgecolor='k', ms=7, label='Significant session (p<0.05)'),
        Line2D([0], [0], marker='o', lw=0, markerfacecolor='white',
               markeredgecolor='lightgray', ms=7, label='Non-significant session'),
    ]
    fig.legend(handles=leg, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[saved] {out_path}')


if not results.empty:
    for modality in ['LFP', 'MUA']:
        plot_summary(results, modality,
                     os.path.join(OUTPUT_DIR, f'vns_cross_subject_{modality.lower()}_boxplot.png'))

    # ── Statistical summary (paired t-test: real vs shuffled) ─────────────────
    rows = []
    for modality in ['LFP', 'MUA']:
        for dtype in ['VNS vs Sham', 'Pulse Width', 'Amplitude']:
            for sub in sorted(results['subject'].unique()):
                sub_df = results[(results['modality'] == modality) &
                                 (results['decode'] == dtype) &
                                 (results['subject'] == sub)].dropna(subset=['accuracy', 'perm_mean'])
                if sub_df.empty:
                    continue
                real = sub_df['accuracy'].values
                shuf = sub_df['perm_mean'].values
                n = len(real)
                if n >= 3:
                    _, pv = stats.ttest_rel(real, shuf, alternative='greater')
                elif n == 2:
                    pv = 0.25 if (real > shuf).all() else 1.0
                else:
                    pv = np.nan
                rows.append(dict(
                    Modality=modality, Decode=dtype, Subject=sub.replace('sub-', ''),
                    N=n,
                    Real_mean=round(real.mean(), 3),
                    Shuf_mean=round(shuf.mean(), 3),
                    Delta=round((real - shuf).mean(), 3),
                    p_value=round(pv, 4) if not np.isnan(pv) else np.nan,
                    Sig=('***' if (not np.isnan(pv) and pv < 0.001) else
                         '**'  if (not np.isnan(pv) and pv < 0.01)  else
                         '*'   if (not np.isnan(pv) and pv < 0.05)  else 'ns')
                ))

    stat_table = pd.DataFrame(rows)
    print('\n=== STATISTICAL SUMMARY (paired t-test: real vs shuffled) ===')
    print(stat_table.to_string(index=False))

    # ── Per-session detail ──────────────────────────────────────────────────
    pivot = results.pivot_table(index=['subject', 'stim', 'modality', 'n_channels', 'n_vns_trains'],
                                columns='decode', values='accuracy').round(3)
    print('\n=== PER-SESSION ACCURACY ===')
    print(pivot.to_string())

    results.to_csv(os.path.join(OUTPUT_DIR, 'vns_cross_subject_lfp_mua_results.csv'), index=False)
    print(f"\n[saved] {os.path.join(OUTPUT_DIR, 'vns_cross_subject_lfp_mua_results.csv')}")
else:
    print('\nNo results to summarize.')

print('\nDone. Output saved to:', OUTPUT_DIR)
