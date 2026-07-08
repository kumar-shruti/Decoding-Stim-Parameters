# VNS Decoding Notebooks

Analyses of vagus nerve stimulation (VNS) electrophysiology recordings: can a
linear decoder tell VNS trials apart from baseline (and from each other) using
spike counts from Kilosort-sorted units?

- **`vns_cross_subject_decoding.ipynb`** — for every recorded
  session, decodes VNS-vs-baseline plus (where enough trials exist)
  amplitude level and pulse width, then aggregates accuracy across
  all subjects/sessions into summary plots and tables.
- **`vns_cross_temporal_decoding.ipynb`** — for one chosen session,
  trains a decoder on each 500 ms time bin and tests it on every other time
  bin, producing a train-time × test-time accuracy matrix (temporal
  generalization) for that session. Each trial's epoch spans as much of the
  surrounding inter-train interval as the data safely allow (half the
  tightest gap between consecutive VNS trains), so the decoding-accuracy
  progression is visible across the full rest period, not just a few
  seconds around onset.

## Data

Raw recordings and Kilosort output are not included in this repo (the full
dataset is >150 GB). Each notebook expects two directories with this layout:

```
kilosort_results/
  sub-<id>/
    stim<N>/
      spike_times.npy        # spike times, in samples (see SAMPLE_RATE)
      spike_clusters.npy     # cluster id per spike
      cluster_KSLabel.tsv    # Kilosort/manual label per cluster ('good', 'mua', 'noise')

derivative/
  sub-<id>/
    stim<N>_noArti.mat       # HDF5-format .mat with 'stim_DAQ/data' and 'time' datasets
```

`kilosort_results/` comes from running Kilosort + manual curation in phy;
`derivative/*_noArti.mat` is the artifact-removed broadband signal recorded
alongside the stimulation DAQ channel used to detect VNS train onsets.

Point the notebooks at your own copy of the data by setting two environment
variables before launching Jupyter (or by editing the defaults directly in
each notebook's first code cell):

```bash
export KILOSORT_RESULTS_DIR=/path/to/kilosort_results
export DERIVATIVE_DIR=/path/to/derivative
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy scikit-learn h5py matplotlib jupyter
jupyter notebook .
```

## Notes

- Both notebooks are self-contained (no shared local modules) — everything
  needed to run one is defined within that notebook.
- `vns_cross_temporal_decoding.ipynb` analyzes a single session, set via the
  `SUBJECT` / `STIM` constants near the top; change these to look at a
  different recording.
- Committed notebook outputs (plots, printed tables) reflect a full run
  against the lab's data and are kept so results are visible without
  needing data access.
