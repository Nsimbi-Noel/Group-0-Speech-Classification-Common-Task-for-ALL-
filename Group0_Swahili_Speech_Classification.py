"""
CSC 63106 - Week 6 Practical - Group 0: Swahili speech (word) classification

Same code as Group0_Swahili_Speech_Classification.ipynb, in "percent format":
every "# %%" line starts a cell, so you can

  * run the whole file:                 python Group0_Swahili_Speech_Classification.py
  * run it cell by cell in VS Code:     click "Run Cell" above any "# %%" line (needs the Jupyter extension)
  * run it cell by cell in Spyder / PyCharm / Jupytext / Colab (File > Upload, then %run or paste cells)

Plain `python` run without a screen (e.g. on a server)?  Set HEADLESS=1 and every plot is saved to outputs/figures/
    PowerShell:  $env:HEADLESS="1"; python Group0_Swahili_Speech_Classification.py
    bash:        HEADLESS=1 python Group0_Swahili_Speech_Classification.py

Read README.md first: it explains how to download the data and what to install.
"""
# %% [setup for running as a plain script]
import os
import matplotlib
HEADLESS = os.environ.get("HEADLESS", "0") == "1"
if HEADLESS:
    matplotlib.use("Agg")                       # no window: save figures instead
import matplotlib.pyplot as _plt
if HEADLESS:
    _count = [0]
    def _save_instead_of_show(*args, **kwargs):
        os.makedirs("outputs/figures", exist_ok=True)
        _count[0] += 1
        _plt.savefig(f"outputs/figures/fig_{_count[0]:02d}.png", dpi=150, bbox_inches="tight")
        _plt.close("all")
    _plt.show = _save_instead_of_show
try:
    display                                     # exists inside Jupyter / VS Code interactive window
except NameError:
    def display(obj):                           # plain python: just print tables / objects
        print(obj)


# %% [markdown]
# # CSC 63106 – Artificial Intelligence
# ## Week 6 Practical · Group 0: Swahili Speech Classification (common task)
#
# **Goal:** build a deep-learning classifier that listens to a short audio clip of a spoken Swahili word and predicts *which word it is*.
#
# ### Roadmap (mirrors the assignment sheet)
#
# | Step | What we do | Assignment item |
# |---|---|---|
# | 1 | Environment, configuration, reproducibility | – |
# | 2 | Load the audio data | *Load audio files* |
# | 3 | Inspect metadata (sample rate, duration, label distribution) | *Inspect metadata* |
# | 4 | Visualise waveform, spectrogram, mel-spectrogram, MFCC | *Visualise with librosa* |
# | 5 | Preprocess: resample, trim, fix length, augment, MFCC, normalise | *Preprocess* |
# | 6 | Build a configurable 1D CNN | *Modeling* |
# | 7 | Train with dropout, early stopping and LR scheduling | *Train 10–15 epochs* |
# | 8 | Evaluate (accuracy, F1, confusion matrix) + PCA / t-SNE embeddings | *Evaluate* |
# | 9 | Explore: sampling rates, spectrogram resolution, architecture ablations | *Exploration* |
# | 10 | Transfer learning: Wav2Vec2 and English → Swahili CNN transfer | *Exploration* |
# | 11 | Extension: speech-to-text → sentiment pipeline | *Extension* |
# | 12 | Report guide | *Deliverables* |
#
# > **How to use this notebook.** Run it top to bottom. By default it runs on a small **synthetic** dataset so you can verify that every cell works in a minute. Then switch `data_source` in Step 1 to `"folder"` or `"hf"` and point it to the real Swahili words dataset, and re-run everything. Use a GPU runtime in Colab (*Runtime → Change runtime type → T4 GPU*).


# %% [markdown]
# ## Step 1 · Setup, configuration and reproducibility
#
# **Why a single config object?** Almost every experiment in Step 9 is "the same pipeline with one setting changed" (sample rate, number of layers, …). Keeping every hyper-parameter in one `Config` dataclass lets us run an experiment with `replace(CFG, kernel=9)` and guarantees nothing else changes by accident.
#
# **Why fix random seeds?** Neural-network training is stochastic (weight initialisation, shuffling, dropout). Fixing the seeds makes results repeatable so differences between experiments are due to our change, not luck.


# %%
# On Google Colab this installs the libraries. Locally (PowerShell) you already did `pip install`, so nothing happens.
import sys, subprocess
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "librosa", "soundfile", "datasets", "transformers", "scikit-learn", "seaborn", "jiwer"])


# %%
import os, random, warnings, time, copy
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa, librosa.display
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# %%
@dataclass
class Config:
    # ---------- data source ----------
    data_source: str = "parquet"        # "parquet" | "folder" | "hf" | "synthetic"
    folder_root: str = "/content/swahili_words"   # folder/<label>/<file>.wav
    parquet_dir: str = "./data"                    # folder with the downloaded *.parquet shards
    per_class: int = 150           # max clips kept per word (keeps classes balanced and CPU training fast)
    hf_name: str = "YOUR_HF_DATASET_NAME"          # e.g. the Swahili words dataset or Common Voice
    hf_config: Optional[str] = None                # e.g. "sw" for Common Voice
    hf_split: str = "train"
    hf_audio_col: str = "audio"
    hf_label_col: str = "text"                     # column holding the word / class
    synthetic_seed: int = 0
    synthetic_classes: int = 10
    max_samples: int = 6000        # cap for speed
    max_classes: int = 10          # keep the N most frequent words
    min_per_class: int = 100       # drop words with too few recordings
    # ---------- audio / features ----------
    sr: int = 16000                # target sample rate
    duration: float = 0.5          # every clip is padded / cropped to this many seconds (99% of trimmed words are <= 0.45 s)
    n_mfcc: int = 40
    n_mels: int = 64
    use_delta: bool = True         # append delta (velocity) features
    aug_copies: int = 2            # augmented copies per *training* clip
    # ---------- model ----------
    n_layers: int = 3
    base_channels: int = 64
    kernel: int = 5
    activation: str = "relu"       # relu | gelu | tanh | leaky_relu
    dropout: float = 0.3
    # ---------- training ----------
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4              # early stopping patience (epochs)
    seed: int = 42

CFG = Config()
set_seed(CFG.seed)
CFG


# %% [markdown]
# ## Step 2 · Load the audio data
#
# The dataset must end up as two things:
#
# * `waves`: a Python list of 1-D NumPy arrays (the raw audio, **kept at its original sample rate** so we can inspect it honestly in Step 3);
# * `df`: a table with one row per clip: `label` (the Swahili word) and `sr` (its original sample rate).
#
# Four loaders are provided:
#
# 0. **`parquet`** – the Hugging Face dataset downloaded to disk as parquet shards (our real Swahili words data). It first counts how often each word occurs, keeps the most frequent words, and loads a *balanced* sample (`per_class` clips per word).
# 1. **`synthetic`** – generates fake "words" (each class is a distinct sequence of three tones plus noise). It is *not* speech; it only lets you check the whole pipeline without downloading anything.
# 2. **`folder`** – any dataset arranged as `root/<word>/<clip>.wav` (very common for word-level datasets; e.g. download the Swahili words dataset, unzip it, and point `folder_root` to it).
# 3. **`hf`** – a Hugging Face dataset with an audio column and a label column.
#
# > **About Common Voice.** Common Voice contains *full sentences*, not isolated words, so there is no ready-made "word" label. If you use it, a practical choice is to treat the **most frequent short sentences / words** as classes (this is what `max_classes` and `min_per_class` do below), or to cut the clips into words with the forced-aligned timestamps. For a clean *word classification* task, the Swahili words speech-text parallel dataset is the better fit.
#
# > **Speaker leakage warning.** If the dataset has a speaker id, split by speaker (not by clip) in Step 5. Otherwise the same voice appears in train and test and accuracy is over-optimistic. Our default stratified-by-label split is fine for the synthetic demo and for datasets without speaker ids.


# %%
def make_synthetic(n_classes=10, n_per_class=120, seed=0):
    # Fake "words": each class = 3 consecutive tones (a tiny "melody"), random duration,
    # random loudness, light pitch jitter, background noise, and mixed sample rates.
    rng = np.random.default_rng(seed)
    melodies = rng.uniform(250, 3000, size=(n_classes, 3))
    waves, rows = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            sr = int(rng.choice([16000, 22050]))
            dur = rng.uniform(0.6, 1.2)
            t = np.arange(int(sr * dur)) / sr
            seg = np.minimum((t / dur * 3).astype(int), 2)
            f = melodies[c][seg] * rng.uniform(0.95, 1.05)
            phase = 2 * np.pi * np.cumsum(f) / sr
            y = np.sin(phase) + 0.4 * np.sin(2 * phase)
            y *= np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.5 * rng.uniform(0.3, 0.9)
            y += rng.normal(0, 0.05, len(y))
            waves.append(y.astype(np.float32)); rows.append(dict(label=f"word_{c}", sr=sr))
    return waves, pd.DataFrame(rows)

def load_folder(root, max_samples, seed):
    paths, labels = [], []
    for lab in sorted(os.listdir(root)):
        d = os.path.join(root, lab)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    paths.append(os.path.join(d, fn)); labels.append(lab)
    order = np.random.default_rng(seed).permutation(len(paths))[:max_samples]
    waves, rows = [], []
    for i in order:
        y, sr = librosa.load(paths[i], sr=None, mono=True)   # sr=None keeps the ORIGINAL rate
        waves.append(y.astype(np.float32)); rows.append(dict(label=labels[i], sr=int(sr)))
    return waves, pd.DataFrame(rows)

def _decode_audio(a):
    # Works with both old (dict) and new (AudioDecoder) versions of `datasets`.
    if isinstance(a, dict):
        arr = np.asarray(a["array"], dtype=np.float32); sr = int(a["sampling_rate"])
    else:
        s = a.get_all_samples(); arr = s.data.numpy().astype(np.float32); sr = int(s.sample_rate)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr, sr

def load_hf(cfg):
    from datasets import load_dataset, Audio
    ds = load_dataset(cfg.hf_name, cfg.hf_config, split=cfg.hf_split)
    ds = ds.cast_column(cfg.hf_audio_col, Audio(sampling_rate=None))  # keep native rate
    if len(ds) > cfg.max_samples:
        ds = ds.shuffle(seed=cfg.seed).select(range(cfg.max_samples))
    feat = ds.features[cfg.hf_label_col]
    names = getattr(feat, "names", None)
    waves, rows = [], []
    for ex in ds:
        arr, sr = _decode_audio(ex[cfg.hf_audio_col])
        lab = ex[cfg.hf_label_col]
        lab = names[lab] if names is not None else str(lab).strip().lower()
        waves.append(arr); rows.append(dict(label=lab, sr=sr))
    return waves, pd.DataFrame(rows)

def keep_top_classes(waves, df, max_classes, min_per_class):
    counts = df["label"].value_counts()
    keep = counts[counts >= min_per_class].index[:max_classes]
    mask = df["label"].isin(keep).values
    waves = [w for w, m in zip(waves, mask) if m]
    return waves, df[mask].reset_index(drop=True)

def load_parquet(cfg):
    # Reads the Hugging Face dataset that was downloaded as parquet shards (columns: audio, text).
    # Pass 1 reads ONLY the text column of every shard (cheap) to find the most frequent words.
    # Pass 2 loads audio for a balanced sample of those words, one shard at a time (low memory).
    import glob, io
    import soundfile as sf
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(cfg.parquet_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .parquet files under {cfg.parquet_dir!r}. Check parquet_dir in the config.")
    names = pq.ParquetFile(files[0]).schema_arrow.names
    for col in (cfg.hf_audio_col, cfg.hf_label_col):
        if col not in names:
            raise KeyError(f"Column {col!r} not found. The shards contain: {names}. Fix hf_audio_col / hf_label_col.")
    parts = []
    for fi, f in enumerate(files):
        t = pq.read_table(f, columns=[cfg.hf_label_col]).to_pandas()
        t.columns = ["label"]; t["file"] = fi; t["row"] = np.arange(len(t))
        parts.append(t)
    meta = pd.concat(parts, ignore_index=True)
    meta["label"] = meta["label"].astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
    meta = meta[~meta["label"].str.startswith("_")]          # drop special labels such as _silence_ / background noise
    counts = meta["label"].value_counts()
    print(f"{len(files)} shards, {len(meta):,} clips, {len(counts):,} distinct words")
    keep = counts[counts >= cfg.min_per_class].index[:cfg.max_classes]
    print("Most frequent words kept:", {w: int(counts[w]) for w in keep})
    sel = pd.concat([g.sample(min(len(g), cfg.per_class), random_state=cfg.seed)
                     for _, g in meta[meta["label"].isin(keep)].groupby("label")])
    waves, rows = [], []
    for fi, g in sel.groupby("file"):
        col = pq.read_table(files[fi], columns=[cfg.hf_audio_col]).column(0)
        for r, lab in zip(g["row"].values, g["label"].values):
            item = col[int(r)].as_py()                      # {'bytes': ..., 'path': ...}
            y, sr = sf.read(io.BytesIO(item["bytes"]), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            waves.append(y); rows.append(dict(label=lab, sr=int(sr)))
    return waves, pd.DataFrame(rows)

def sample_cache_path(cfg):
    tag = os.path.basename(os.path.normpath(cfg.parquet_dir))
    prefix = "swahili_sample" if tag == "data" else f"sample_{tag}"
    return f"{prefix}_{cfg.max_classes}w_{cfg.per_class}n_{cfg.min_per_class}m_s{cfg.seed}.pkl"

def load_audio(cfg):
    if cfg.data_source == "synthetic":
        waves, df = make_synthetic(cfg.synthetic_classes, seed=cfg.synthetic_seed)
    elif cfg.data_source == "parquet":
        import pickle
        cp = sample_cache_path(cfg)
        if os.path.exists(cp):                       # small cached sample -> no need for the 3.9 GB dataset
            with open(cp, "rb") as fh:
                waves, df = pickle.load(fh)
            print(f"Loaded cached sample from {cp}")
        else:
            waves, df = load_parquet(cfg)
            with open(cp, "wb") as fh:
                pickle.dump((waves, df), fh)
            print(f"Saved the sample to {cp} (copy this small file to Colab to skip the big dataset)")
    elif cfg.data_source == "folder":
        waves, df = load_folder(cfg.folder_root, cfg.max_samples, cfg.seed)
    elif cfg.data_source == "hf":
        waves, df = load_hf(cfg)
    else:
        raise ValueError(cfg.data_source)
    waves, df = keep_top_classes(waves, df, cfg.max_classes, cfg.min_per_class)
    df["n_samples"] = [len(w) for w in waves]
    df["duration"] = df["n_samples"] / df["sr"]
    return waves, df

waves, df = load_audio(CFG)
print(f"{len(waves)} clips, {df['label'].nunique()} classes")
df.head()


# %% [markdown]
# ## Step 3 · Inspect metadata
#
# Before modelling, always answer three questions about audio data:
#
# 1. **Sample rate** – how many measurements per second? Models need a *consistent* rate. Mixed rates must be resampled to one (Step 5). The Nyquist theorem says a rate `sr` can only represent frequencies up to `sr/2` (16 kHz → 8 kHz, which covers speech).
# 2. **Duration** – decides the fixed input length we pad/crop to. Too short crops cut off words; too long wastes computation.
# 3. **Label distribution** – a strongly imbalanced dataset makes *accuracy* misleading (a model that always says the biggest class looks good). That's why we also report **macro-F1** later.


# %%
print("Sample-rate counts:\n", df["sr"].value_counts(), "\n")
print("Duration (seconds):\n", df["duration"].describe().round(3), "\n")
counts = df["label"].value_counts()
print(f"Classes: {len(counts)} | min/median/max clips per class: {counts.min()}/{int(counts.median())}/{counts.max()}")

fig, ax = plt.subplots(1, 3, figsize=(17, 4))
sns.countplot(x="sr", data=df, ax=ax[0]); ax[0].set_title("Original sample rates")
sns.histplot(df["duration"], bins=30, ax=ax[1], kde=True); ax[1].set_title("Clip duration (s)"); ax[1].set_xlabel("seconds")
order = counts.index
sns.countplot(y="label", data=df, order=order, ax=ax[2]); ax[2].set_title("Label distribution")
plt.tight_layout(); plt.show()


# %%
# Length of the actual speech after trimming silence (this is what decides the window length)
L = np.array([len(librosa.effects.trim(w, top_db=30)[0]) / 16000 for w in waves])
print("Trimmed word length (seconds):")
print(pd.Series(L).quantile([0.5, 0.9, 0.95, 0.99, 1.0]).round(3))


# %% [markdown]
# **How to read this.** Look for (a) more than one sample rate → resampling is required; (b) the typical duration → choose `duration` in the config (1 s here; raise it if most clips are longer); (c) bars of very different heights → class imbalance, so trust macro-F1 more than accuracy. The trimmed-length table is what we use to choose `duration`: after silence is removed, 99% of the words are about 0.45 s or shorter, so a 0.5 s window keeps nearly every word whole and wastes far less input on padding than a 1 s window would.


# %% [markdown]
# ## Step 4 · Visualise waveform, spectrogram, mel-spectrogram and MFCC
#
# Four views of the *same* word, each one a more useful representation for a neural network:
#
# * **Waveform** – air-pressure vs. time. Hard to learn from directly (very long, phase-sensitive).
# * **Spectrogram (STFT, in dB)** – cut the signal into short overlapping windows (25 ms, every 10 ms), compute the frequency content of each. The x-axis is time, the y-axis is frequency, colour is energy.
# * **Mel-spectrogram** – the same, but the frequency axis is warped to the **mel scale**, which spaces frequencies roughly like human hearing does (fine detail at low frequencies, coarse at high).
# * **MFCC** – a compact summary of the mel-spectrogram (a cosine transform keeps only the smooth "envelope" of the spectrum, which is what distinguishes sounds/vowels/consonants and largely removes the speaker's pitch fine-structure). This is our input feature.


# %%
def show_views(y, sr, title=""):
    n_fft, hop = 1024, int(sr * 0.010)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)), ref=np.max)
    M = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64), ref=np.max)
    C = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop)
    fig, ax = plt.subplots(2, 2, figsize=(14, 7))
    librosa.display.waveshow(y, sr=sr, ax=ax[0, 0]); ax[0, 0].set_title("Waveform")
    im = librosa.display.specshow(S, sr=sr, hop_length=hop, x_axis="time", y_axis="hz", ax=ax[0, 1]); ax[0, 1].set_title("Spectrogram (dB)"); fig.colorbar(im, ax=ax[0, 1])
    im = librosa.display.specshow(M, sr=sr, hop_length=hop, x_axis="time", y_axis="mel", ax=ax[1, 0]); ax[1, 0].set_title("Mel-spectrogram (dB)"); fig.colorbar(im, ax=ax[1, 0])
    im = librosa.display.specshow(C, sr=sr, hop_length=hop, x_axis="time", ax=ax[1, 1]); ax[1, 1].set_title("MFCC (20 coefficients)"); fig.colorbar(im, ax=ax[1, 1])
    fig.suptitle(title, fontsize=14); plt.tight_layout(); plt.show()

# Show two different words side by side in the sense of "do they look different?"
for lab in df["label"].unique()[:2]:
    i = df.index[df["label"] == lab][0]
    show_views(waves[i], int(df.loc[i, "sr"]), title=f"Word: {lab}")


# %% [markdown]
# **How to read this.** Different words should produce visibly different patterns in the mel-spectrogram/MFCC (different energy bands at different times). If two classes look identical to you, they will be hard for the model too. Listen to a clip with `IPython.display.Audio(waves[i], rate=sr)` when working with real data.


# %%
# Optional: listen to a clip (works in Colab / Jupyter)
import IPython.display as ipd
i = 0
ipd.Audio(waves[i], rate=int(df.loc[i, "sr"]))


# %% [markdown]
# ## Step 5 · Preprocessing: resample, trim, fix length, augment, extract MFCC, normalise
#
# Each operation and the reason for it:
#
# | Operation | Why |
# |---|---|
# | **Split first** (70 % train / 15 % val / 15 % test, stratified) | Prevents *data leakage*: augmentation and normalisation statistics must come from the training set only, or the test score is inflated. |
# | **Resample** to one `sr` | The network's input size depends on sample rate; all clips must share it. |
# | **Trim silence** | Removes leading/trailing silence so the word is aligned in the window. |
# | **Peak-normalise amplitude** | Removes loudness differences (microphone distance, volume). |
# | **Pad / crop to fixed length** | CNNs need equal-size input batches. |
# | **Augmentation** (noise, pitch shift, time shift) – *training set only* | Creates plausible variations (noisy room, different speaker pitch, word starting slightly earlier/later), which reduces over-fitting. Val/test stay untouched so they measure real-world performance. |
# | **MFCC (+ deltas)** | Compact, speech-oriented features. Deltas capture how coefficients change over time. 10 ms hop → ~100 frames per second. |
# | **Standardise per feature** using training mean/std | Keeps feature scales comparable so the optimiser trains faster and more stably. |
#
# The 1D CNN will treat the **MFCC coefficients as input channels** and **time as the sequence axis**: input shape `(batch, n_features, n_frames)`.


# %%
def preprocess_wave(y, sr, cfg):
    if sr != cfg.sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sr)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) == 0:
        y = np.zeros(1, dtype=np.float32)
    y = y / (np.max(np.abs(y)) + 1e-9)
    n = int(cfg.sr * cfg.duration)
    if len(y) >= n:                       # centre-crop
        s = (len(y) - n) // 2; y = y[s:s + n]
    else:                                 # centre-pad with zeros
        pad = n - len(y); y = np.pad(y, (pad // 2, pad - pad // 2))
    return y.astype(np.float32)

# ---------- augmentations (applied to the training set only) ----------
def add_noise(y, rng, snr_db_range=(10, 30)):
    snr = rng.uniform(*snr_db_range)                       # signal-to-noise ratio in dB
    p_noise = (np.mean(y ** 2) + 1e-12) / (10 ** (snr / 10))
    return y + rng.normal(0, np.sqrt(p_noise), len(y)).astype(np.float32)

def pitch_shift(y, sr, rng, max_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=rng.uniform(-max_steps, max_steps))

def time_shift(y, rng, max_frac=0.15):
    return np.roll(y, int(rng.uniform(-max_frac, max_frac) * len(y)))

def augment(y, sr, rng):
    if rng.random() < 0.7: y = time_shift(y, rng)
    if rng.random() < 0.5: y = pitch_shift(y, sr, rng)
    if rng.random() < 0.7: y = add_noise(y, rng)
    return np.clip(y, -1, 1).astype(np.float32)

# ---------- feature extraction ----------
def extract_features(y, cfg):
    hop = int(cfg.sr * 0.010)                                   # 10 ms step
    n_fft = 2 ** int(np.ceil(np.log2(cfg.sr * 0.025)))          # ~25 ms window, power of 2
    m = librosa.feature.mfcc(y=y, sr=cfg.sr, n_mfcc=cfg.n_mfcc, n_fft=n_fft,
                             hop_length=hop, n_mels=cfg.n_mels)
    if cfg.use_delta:
        m = np.vstack([m, librosa.feature.delta(m)])
    return m.astype(np.float32)                                 # (n_features, n_frames)

# ---------- full dataset builder ----------
def build_data(waves, df, cfg):
    rng = np.random.default_rng(cfg.seed)
    le = LabelEncoder().fit(df["label"])
    y_all = le.transform(df["label"])
    srs = df["sr"].values
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y_all, random_state=cfg.seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y_all[tmp], random_state=cfg.seed)

    clean = {i: preprocess_wave(waves[i], srs[i], cfg) for i in idx}

    def feats(ids, aug=False):
        X, Y = [], []
        for i in ids:
            X.append(extract_features(clean[i], cfg)); Y.append(y_all[i])
            if aug:
                for _ in range(cfg.aug_copies):
                    X.append(extract_features(augment(clean[i], cfg.sr, rng), cfg)); Y.append(y_all[i])
        return np.stack(X), np.array(Y)

    Xtr, ytr = feats(tr, aug=True); Xva, yva = feats(va); Xte, yte = feats(te)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)                   # statistics from TRAIN only
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    norm = lambda X: ((X - mu) / sd).astype(np.float32)
    return dict(mu=mu, sd=sd, Xtr=norm(Xtr), ytr=ytr, Xva=norm(Xva), yva=yva, Xte=norm(Xte), yte=yte,
                le=le, classes=list(le.classes_),
                wav_tr=[clean[i] for i in tr], wav_va=[clean[i] for i in va], wav_te=[clean[i] for i in te],
                y_wav_tr=y_all[tr], y_wav_va=y_all[va], y_wav_te=y_all[te])

_DATA_CACHE = {}
def get_data(cfg, waves, df):
    key = (cfg.sr, cfg.n_mfcc, cfg.n_mels, cfg.use_delta, cfg.aug_copies, cfg.duration, cfg.seed, cfg.data_source, cfg.synthetic_seed, cfg.parquet_dir, cfg.max_classes, cfg.per_class)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = build_data(waves, df, cfg)
    return _DATA_CACHE[key]

t0 = time.time()
data = get_data(CFG, waves, df)
print(f"Feature extraction took {time.time() - t0:.1f}s")
for k in ["Xtr", "Xva", "Xte"]:
    print(f"{k}: {data[k].shape}   (samples, features, frames)")
print("Classes:", data["classes"])


# %% [markdown]
# **Sanity check – what did augmentation do?** Plot one clean clip next to augmented versions of it. If the augmented MFCCs look wildly different (or the word is destroyed), the augmentation is too aggressive.


# %%
rng = np.random.default_rng(0)
y0 = data["wav_tr"][0]
variants = {"clean": y0, "+ noise": add_noise(y0, rng, (5, 10)),
            "+ pitch shift": pitch_shift(y0, CFG.sr, rng), "+ time shift": time_shift(y0, rng, 0.3)}
fig, ax = plt.subplots(2, 4, figsize=(17, 6))
for j, (name, y) in enumerate(variants.items()):
    librosa.display.waveshow(y, sr=CFG.sr, ax=ax[0, j]); ax[0, j].set_title(name)
    librosa.display.specshow(extract_features(y, CFG)[:CFG.n_mfcc], x_axis="time", sr=CFG.sr, hop_length=int(CFG.sr * 0.01), ax=ax[1, j])
plt.tight_layout(); plt.show()


# %% [markdown]
# ## Step 6 · The model: a configurable 1D CNN
#
# **Why a 1D convolution?** Our input is a sequence of MFCC frames. A 1D convolution slides a small window (the *kernel*) along the **time axis**, looking at all coefficients at once. Early layers learn short-range patterns (a few tens of milliseconds: bursts, vowel onsets); deeper layers, whose receptive field grows because of pooling, learn longer patterns (syllables, whole words).
#
# **Architecture** (every block is repeated `n_layers` times):
#
# ```
# Conv1d(kernel) → BatchNorm → activation → MaxPool(2) → Dropout
#             ↓  (after the blocks)
# Global average pooling over time  → Linear(128) → activation → Dropout → Linear(n_classes)
# ```
#
# * **BatchNorm** stabilises and speeds up training.
# * **MaxPool(2)** halves the time length and enlarges the receptive field.
# * **Global average pooling** makes the model independent of the exact number of frames and greatly reduces parameters (less over-fitting than flattening).
# * **Dropout** randomly switches off neurons during training so the network can't rely on any single one – a regulariser.
# * The 128-d layer before the output is the **embedding** we visualise in Step 8.
#
# The knobs we will experiment with: `n_layers`, `kernel`, `activation`, `dropout`.


# %%
ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}

class CNN1D(nn.Module):
    def __init__(self, in_ch, n_classes, n_layers=3, base=64, kernel=5, act="relu", dropout=0.3):
        super().__init__()
        blocks, c = [], in_ch
        for i in range(n_layers):
            out = min(base * 2 ** i, 256)
            blocks += [nn.Conv1d(c, out, kernel, padding=kernel // 2), nn.BatchNorm1d(out),
                       ACTS[act](), nn.MaxPool1d(2), nn.Dropout(dropout)]
            c = out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embed = nn.Sequential(nn.Linear(c, 128), ACTS[act](), nn.Dropout(dropout))
        self.head = nn.Linear(128, n_classes)

    def forward(self, x, return_embedding=False):
        h = self.pool(self.features(x)).squeeze(-1)
        e = self.embed(h)
        out = self.head(e)
        return (out, e) if return_embedding else out

def build_cnn(cfg, in_ch, n_classes):
    return CNN1D(in_ch, n_classes, cfg.n_layers, cfg.base_channels, cfg.kernel, cfg.activation, cfg.dropout)

model = build_cnn(CFG, data["Xtr"].shape[1], len(data["classes"])).to(DEVICE)
print(model)
print(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# %% [markdown]
# ## Step 7 · Training with dropout, early stopping and learning-rate scheduling
#
# * **Loss:** cross-entropy – the standard loss for multi-class classification.
# * **Optimiser:** AdamW (Adam with proper weight decay – a mild L2 penalty that also fights over-fitting).
# * **Learning-rate scheduler (`ReduceLROnPlateau`):** if the validation loss has not improved for 2 epochs, halve the learning rate. Big steps early, fine steps later.
# * **Early stopping:** if the validation loss does not improve for `patience` epochs, stop and **restore the best weights**. This prevents memorising the training set.
# * **Epochs:** up to 15 (assignment: 10–15). Early stopping may end sooner.
#
# **How to read the curves.** Training loss always goes down. If validation loss starts *rising* while training loss keeps falling → over-fitting (more dropout / augmentation / earlier stopping). If both stay high → under-fitting (bigger model, more epochs, better features).


# %%
def make_loaders(data, cfg):
    mk = lambda X, y, shuffle: DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long)),
                                          batch_size=cfg.batch_size, shuffle=shuffle)
    return {"train": mk(data["Xtr"], data["ytr"], True),
            "val": mk(data["Xva"], data["yva"], False),
            "test": mk(data["Xte"], data["yte"], False)}

def run_epoch(model, loader, crit, opt=None):
    training = opt is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = crit(out, yb)
            if training:
                opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n

def train_model(model, loaders, cfg, epochs=None, lr=None, verbose=True):
    epochs = epochs or cfg.epochs
    lr = lr or cfg.lr
    model.to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "lr"]}
    best_loss, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(model, loaders["train"], crit, opt)
        vl, va = run_epoch(model, loaders["val"], crit)
        sched.step(vl)
        cur_lr = opt.param_groups[0]["lr"]
        for k, v in zip(hist, [tl, vl, ta, va, cur_lr]):
            hist[k].append(v)
        if verbose:
            print(f"epoch {ep:2d} | train loss {tl:.4f} acc {ta:.3f} | val loss {vl:.4f} acc {va:.3f} | lr {cur_lr:.2e}")
        if vl < best_loss - 1e-4:
            best_loss, bad = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose: print(f"Early stopping at epoch {ep} (best val loss {best_loss:.4f})")
                break
    model.load_state_dict(best_state)            # restore the best weights
    return hist

def plot_history(hist, title=""):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="validation")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="validation")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    ax[2].plot(hist["lr"]); ax[2].set_title("Learning rate (scheduler)"); ax[2].set_xlabel("epoch"); ax[2].set_yscale("log")
    fig.suptitle(title); plt.tight_layout(); plt.show()

set_seed(CFG.seed)
loaders = make_loaders(data, CFG)
history = train_model(model, loaders, CFG)
plot_history(history, "Baseline 1D CNN")


# %% [markdown]
# ## Step 8 · Evaluation and embedding visualisation
#
# We evaluate **once**, on the held-out **test set** (never used for training or model selection):
#
# * **Accuracy** – fraction correct. Simple, but misleading under class imbalance.
# * **F1-score** – harmonic mean of precision (of the clips predicted as word *w*, how many really were *w*) and recall (of the real *w* clips, how many we found). **Macro-F1** averages F1 over classes equally, so rare words matter as much as common ones.
# * **Confusion matrix** – rows = true word, columns = predicted word. The diagonal is correct; off-diagonal cells show *which* words get confused (usually similar-sounding ones).
# * **Embeddings (PCA / t-SNE)** – we take the 128-d vector before the output layer and squeeze it into 2-D. If the network learned well, clips of the same word form tight clusters. PCA is linear and preserves global structure; t-SNE is non-linear and reveals local clusters (but distances *between* clusters are not meaningful).


# %%
@torch.no_grad()
def predict(model, X, bs=256):
    model.eval()
    probs, embs = [], []
    for i in range(0, len(X), bs):
        out, e = model(torch.tensor(X[i:i + bs]).to(DEVICE), return_embedding=True)
        probs.append(torch.softmax(out, 1).cpu().numpy()); embs.append(e.cpu().numpy())
    probs, embs = np.concatenate(probs), np.concatenate(embs)
    return probs.argmax(1), probs, embs

def evaluate(model, data, show=True):
    y_pred, probs, emb = predict(model, data["Xte"])
    y_true = data["yte"]
    res = dict(acc=accuracy_score(y_true, y_pred),
               f1_macro=f1_score(y_true, y_pred, average="macro"),
               f1_weighted=f1_score(y_true, y_pred, average="weighted"),
               y_true=y_true, y_pred=y_pred, emb=emb)
    if show:
        print(f"Test accuracy : {res['acc']:.4f}")
        print(f"Macro F1      : {res['f1_macro']:.4f}")
        print(f"Weighted F1   : {res['f1_weighted']:.4f}\n")
        print(classification_report(y_true, y_pred, target_names=data["classes"], digits=3, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        n = len(data["classes"])
        plt.figure(figsize=(max(6, n * 0.5), max(5, n * 0.45)))
        sns.heatmap(cm, annot=n <= 15, fmt=".2f", cmap="Blues", xticklabels=data["classes"], yticklabels=data["classes"])
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix (row-normalised)"); plt.tight_layout(); plt.show()
    return res

res = evaluate(model, data)


# %%
def plot_embeddings(emb, y, classes, title=""):
    pca = PCA(n_components=2).fit(emb)
    z_pca = pca.transform(emb)
    z_tsne = TSNE(n_components=2, perplexity=max(5, min(30, len(emb) // 4)), init="pca", random_state=0).fit_transform(emb)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for k, (z, name) in enumerate([(z_pca, f"PCA (explains {pca.explained_variance_ratio_.sum():.0%} of variance)"), (z_tsne, "t-SNE")]):
        sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=[classes[i] for i in y], palette="tab20", s=35, ax=ax[k], legend=(k == 1))
        ax[k].set_title(name)
    if ax[1].get_legend(): ax[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle(title); plt.tight_layout(); plt.show()

plot_embeddings(res["emb"], res["y_true"], data["classes"], "Test-set embeddings of the baseline CNN")


# %%
# Listen to the clips the model got wrong (useful for the error analysis in the report)
import IPython.display as ipd
wrong = np.where(res["y_true"] != res["y_pred"])[0]
for i in wrong:
    print(f"true = {data['classes'][res['y_true'][i]]}, predicted = {data['classes'][res['y_pred'][i]]}")
    display(ipd.Audio(data["wav_te"][i], rate=CFG.sr))


# %% [markdown]
# **What to write in the report here:** which words are confused with each other and *why* (similar phonemes? short words? speaker/noise variation?), whether clusters are tight/separate, and whether train vs. validation curves show over-fitting.


# %% [markdown]
# ## Step 9 · Exploration experiments
#
# We now change **one factor at a time** (everything else = baseline `CFG`) and record test accuracy and macro-F1. Changing one thing at a time is what lets us attribute a difference to that factor (an *ablation study*). Each run is shorter (8 epochs) to keep the total time reasonable.
#
# > **Caveat.** One run per setting has random noise. Differences of 1–2 points may not be meaningful. If you have time, repeat each experiment with 3 seeds and report mean ± std.
#
# ### 9a · Sampling rate and spectrogram resolution
# * **Sampling rate** (4 / 8 / 16 kHz): the dataset is natively 16 kHz, so we *down*-sample (up-sampling would add no information). Lower rates discard high-frequency detail (8 kHz keeps only up to 4 kHz, which can hurt fricatives like *s, sh*) but make the input smaller and faster.
# * **Spectrogram resolution:** `n_mels` (frequency resolution) and `n_mfcc` (how many cepstral coefficients we keep). More = more detail but more risk of noise/over-fitting.


# %%
def run_experiment(name, cfg_e, epochs=8):
    d = get_data(cfg_e, waves, df)
    set_seed(cfg_e.seed)
    m = build_cnn(cfg_e, d["Xtr"].shape[1], len(d["classes"]))
    t0 = time.time()
    hist = train_model(m, make_loaders(d, cfg_e), cfg_e, epochs=epochs, verbose=False)
    r = evaluate(m, d, show=False)
    return dict(experiment=name, test_acc=round(r["acc"], 4), macro_f1=round(r["f1_macro"], 4),
                best_val_loss=round(min(hist["val_loss"]), 4), epochs_run=len(hist["val_loss"]),
                params=sum(p.numel() for p in m.parameters()), input_shape=str(tuple(d["Xtr"].shape[1:])),
                seconds=round(time.time() - t0, 1))

fast = replace(CFG, aug_copies=1)    # fewer augmented copies to speed up the sweeps
results = []
for sr in [4000, 8000, 16000]:
    results.append(run_experiment(f"sample_rate={sr}", replace(fast, sr=sr)))
for n_mels, n_mfcc in [(32, 20), (64, 40), (128, 40)]:
    results.append(run_experiment(f"n_mels={n_mels}, n_mfcc={n_mfcc}", replace(fast, n_mels=n_mels, n_mfcc=n_mfcc)))
res_a = pd.DataFrame(results); res_a


# %% [markdown]
# ### 9b · Architecture: number of conv layers, kernel size, activation, regularisation
# * **Layers:** more layers = larger receptive field and capacity, but harder to train/over-fit on small data.
# * **Kernel size:** small kernels (3) see short time spans; large kernels (9) see more context per layer.
# * **Activation:** ReLU (default, simple), LeakyReLU (avoids "dead" units), GELU (smooth, used in transformers), tanh (saturating – usually slower).
# * **Augmentation / dropout ablations:** show whether our regularisation actually helps.


# %%
results = []
for L in [2, 3, 4]:
    results.append(run_experiment(f"n_layers={L}", replace(fast, n_layers=L)))
for k in [3, 5, 9]:
    results.append(run_experiment(f"kernel={k}", replace(fast, kernel=k)))
for a in ["relu", "leaky_relu", "gelu", "tanh"]:
    results.append(run_experiment(f"activation={a}", replace(fast, activation=a)))
for p in [0.0, 0.3, 0.5]:
    results.append(run_experiment(f"dropout={p}", replace(fast, dropout=p)))
results.append(run_experiment("no augmentation", replace(fast, aug_copies=0)))
res_b = pd.DataFrame(results); res_b


# %% [markdown]
# ### 9c · How much is noise? Repeat the key comparison with 3 seeds
# The test set has only 225 clips, so one clip is worth about 0.44 accuracy points. Differences of 1–2 points in the tables above are therefore only 2–5 clips and can easily be luck. A fairer comparison repeats a run with different seeds (each seed changes the train/val/test split and the initial weights) and reports **mean ± standard deviation**. We do this for the most surprising result: whether augmentation helps.


# %%
def multi_seed(label, cfg_base, seeds=(0, 1, 2), epochs=15):
    rows = []
    for s in seeds:
        cfg_s = replace(cfg_base, seed=s)
        d = get_data(cfg_s, waves, df)
        set_seed(s)
        m = build_cnn(cfg_s, d["Xtr"].shape[1], len(d["classes"]))
        train_model(m, make_loaders(d, cfg_s), cfg_s, epochs=epochs, verbose=False)
        r = evaluate(m, d, show=False)
        rows.append(dict(setting=label, seed=s, test_acc=r["acc"], macro_f1=r["f1_macro"]))
    return rows

rows = multi_seed("with augmentation (2 copies)", CFG) + multi_seed("no augmentation", replace(CFG, aug_copies=0))
seeds_df = pd.DataFrame(rows)
summary = seeds_df.groupby("setting")[["test_acc", "macro_f1"]].agg(["mean", "std"]).round(4)
display(seeds_df.round(4)); display(summary)


# %% [markdown]
# ### 9d · Robustness to noise
# A model that scores well on clean test clips can still fail in a noisy room. Here we add white noise to the **test** clips at several signal-to-noise ratios (SNR, lower = noisier; measured over the whole window) and compare the model trained *with* augmentation against one trained *without*. This tests the real purpose of augmentation: robustness, not clean-data accuracy.


# %%
def eval_with_noise(model, d, cfg, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    for w in d["wav_te"]:
        w2 = w if snr_db is None else np.clip(add_noise(w, rng, (snr_db, snr_db)), -1, 1)
        X.append(extract_features(w2, cfg))
    X = ((np.stack(X) - d["mu"]) / d["sd"]).astype(np.float32)
    y_pred, _, _ = predict(model, X)
    return accuracy_score(d["yte"], y_pred), f1_score(d["yte"], y_pred, average="macro")

d_noaug = get_data(replace(CFG, aug_copies=0), waves, df)
set_seed(CFG.seed)
m_noaug = build_cnn(CFG, d_noaug["Xtr"].shape[1], len(d_noaug["classes"]))
train_model(m_noaug, make_loaders(d_noaug, CFG), CFG, verbose=False)

snr_levels = [None, 30, 20, 10, 5, 0]
rows = []
for name, m, d in [("trained with augmentation", model, data), ("trained without augmentation", m_noaug, d_noaug)]:
    for snr in snr_levels:
        acc, f1 = eval_with_noise(m, d, CFG, snr)
        rows.append(dict(model=name, snr="clean" if snr is None else f"{snr} dB", test_acc=acc, macro_f1=f1))
noise_df = pd.DataFrame(rows)
order = ["clean"] + [f"{s} dB" for s in snr_levels[1:]]
display(noise_df.pivot(index="snr", columns="model", values="macro_f1").loc[order].round(3))
plt.figure(figsize=(7, 4))
sns.lineplot(data=noise_df, x="snr", y="macro_f1", hue="model", marker="o", sort=False)
plt.title("Robustness to added noise (test set)"); plt.xlabel("noise level (SNR)"); plt.ylabel("macro-F1")
plt.tight_layout(); plt.show()


# %%
all_res = pd.concat([res_a, res_b], ignore_index=True)
fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(all_res))))
sns.barplot(data=all_res, y="experiment", x="macro_f1", color="steelblue", ax=ax)
ax.set_xlim(max(0, all_res["macro_f1"].min() - 0.05), 1.0); ax.set_title("Macro-F1 on the test set per experiment")
plt.tight_layout(); plt.show()
all_res.sort_values("macro_f1", ascending=False).head(5)


# %% [markdown]
# **What to write in the report:** the best and worst settings, the trend (e.g. "16 kHz ≈ 22 kHz > 8 kHz, so speech energy above 4 kHz matters"), and the cost side (parameters, time). Mention that single-seed differences within ~1–2 points are within noise.


# %% [markdown]
# ## Step 10 · Transfer learning
#
# **Idea:** a model that has already learned what speech sounds look like (from a big dataset, possibly in another language) needs far fewer Swahili examples. Two experiments:
#
# * **10a – Fine-tune Wav2Vec2** (a large self-supervised speech Transformer). The raw waveform goes in; we replace its head with a classifier for our words. Needs a GPU.
# * **10b – English → Swahili transfer with our own CNN.** Pre-train the CNN on English words, then copy its convolutional layers into a new model and fine-tune on Swahili. We compare three settings **with limited Swahili data (10 %, 25 %, 100 %)**, because that is where transfer learning helps most.


# %% [markdown]
# ### 10a · Fine-tune Wav2Vec2
#
# * `facebook/wav2vec2-base` was pre-trained on **English** audio (so it is already a transfer from English).
# * `facebook/wav2vec2-xls-r-300m` was pre-trained on **128 languages including Swahili** – usually better for our task, but bigger (needs more GPU memory; lower the batch size if you run out).
# * We **freeze the CNN feature encoder** (standard practice: it is already good and freezing saves memory) and fine-tune the Transformer layers with a *small* learning rate (3e-5) so we do not destroy pre-trained knowledge.


# %%
RUN_WAV2VEC2 = torch.cuda.is_available()     # needs a GPU; set True/False manually if you like
W2V_NAME = "facebook/wav2vec2-base"          # or "facebook/wav2vec2-xls-r-300m"

class W2VWrap(nn.Module):
    # Wrap the HF model so it has the same interface as our CNN (logits, optional embedding).
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x, return_embedding=False):
        out = self.m(x, output_hidden_states=True)
        if return_embedding:
            return out.logits, out.hidden_states[-1].mean(dim=1)
        return out.logits

if RUN_WAV2VEC2:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    fe = AutoFeatureExtractor.from_pretrained(W2V_NAME)
    prep = lambda ws: fe(ws, sampling_rate=CFG.sr, return_tensors="np", padding=True)["input_values"].astype(np.float32)
    w2v_data = dict(Xtr=prep(data["wav_tr"]), ytr=data["y_wav_tr"], Xva=prep(data["wav_va"]), yva=data["y_wav_va"],
                    Xte=prep(data["wav_te"]), yte=data["y_wav_te"], classes=data["classes"])
    hf_model = AutoModelForAudioClassification.from_pretrained(
        W2V_NAME, num_labels=len(data["classes"]),
        label2id={c: i for i, c in enumerate(data["classes"])}, id2label=dict(enumerate(data["classes"])))
    hf_model.freeze_feature_encoder()
    w2v_cfg = replace(CFG, batch_size=16, lr=3e-5, epochs=5, patience=2)
    set_seed(CFG.seed)
    w2v = W2VWrap(hf_model)
    w2v_hist = train_model(w2v, make_loaders(w2v_data, w2v_cfg), w2v_cfg)
    plot_history(w2v_hist, "Wav2Vec2 fine-tuning")
    w2v_res = evaluate(w2v, w2v_data)
    print(f"CNN baseline  : acc {res['acc']:.4f}, macro-F1 {res['f1_macro']:.4f}")
    print(f"Wav2Vec2      : acc {w2v_res['acc']:.4f}, macro-F1 {w2v_res['f1_macro']:.4f}")
else:
    print("No GPU detected - skipping Wav2Vec2 fine-tuning. In Colab: Runtime > Change runtime type > T4 GPU.")


# %% [markdown]
# ### 10b · English → Swahili transfer with the 1D CNN
#
# Procedure:
# 1. Load an **English** speech dataset with *word-level* labels. We use **Google Speech Commands v0.02** (one-second recordings of single English words, re-hosted as parquet; download command in the README). English Common Voice contains full sentences and needs a login, so it does not give clean word labels. If `./data_en` is missing, the cell falls back to synthetic tones and prints a warning – those numbers must not be reported.
# 2. Train the CNN on English (the network learns generic speech/acoustic building blocks).
# 3. For Swahili, compare **three** models trained on only a fraction of the Swahili training data:
#    * **scratch** – random initial weights (no transfer);
#    * **frozen** – copy English conv layers, *freeze* them, train only the new classifier (feature extraction);
#    * **fine-tune** – copy English conv layers, train everything with a smaller learning rate.
#
# The output layer is never copied because the English and Swahili label sets differ.


# %%
# ---- English dataset configuration ----
# REAL English words: download Speech Commands once (see the README), then this cell finds ./data_en automatically:
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('pollen-robotics/speech-commands-v0.02', repo_type='dataset', local_dir='./data_en')"
import glob
if glob.glob(os.path.join("./data_en", "**", "*.parquet"), recursive=True):
    EN_CFG = replace(CFG, data_source="parquet", parquet_dir="./data_en", hf_label_col="label",
                     max_classes=20, per_class=200, min_per_class=200)
    EN_REAL = True
else:
    EN_CFG = replace(CFG, data_source="synthetic", synthetic_seed=999, synthetic_classes=12)
    EN_REAL = False
    print("=" * 78)
    print("WARNING: ./data_en not found, so the 'English' data below is SYNTHETIC TONES, not speech.")
    print("The transfer results are then only a pipeline check. Do NOT report them as English -> Swahili.")
    print("=" * 78)

en_waves, en_df = load_audio(EN_CFG)
en_data = get_data(EN_CFG, en_waves, en_df)
set_seed(CFG.seed)
en_model = build_cnn(CFG, en_data["Xtr"].shape[1], len(en_data["classes"]))
en_hist = train_model(en_model, make_loaders(en_data, CFG), CFG, verbose=False)
en_res = evaluate(en_model, en_data, show=False)
print(f"English pre-training done: test acc {en_res['acc']:.3f}, macro-F1 {en_res['f1_macro']:.3f}")


# %%
def subset_train(d, frac, seed=0):
    if frac >= 1.0:
        return d
    idx, _ = train_test_split(np.arange(len(d["ytr"])), train_size=frac, stratify=d["ytr"], random_state=seed)
    return {**d, "Xtr": d["Xtr"][idx], "ytr": d["ytr"][idx]}

def make_transfer_model(mode):
    m = build_cnn(CFG, data["Xtr"].shape[1], len(data["classes"]))
    if mode in ("frozen", "finetune"):
        state = {k: v for k, v in en_model.state_dict().items() if not k.startswith("head")}
        m.load_state_dict(state, strict=False)              # copy everything except the output layer
    if mode == "frozen":
        for p in m.features.parameters():
            p.requires_grad = False
    return m

rows = []
for frac in [0.1, 0.25, 1.0]:
    d_sub = subset_train(data, frac)
    for mode in ["scratch", "frozen", "finetune"]:
        set_seed(CFG.seed)
        m = make_transfer_model(mode)
        lr = CFG.lr / 3 if mode == "finetune" else CFG.lr
        train_model(m, make_loaders(d_sub, CFG), CFG, lr=lr, verbose=False)
        r = evaluate(m, data, show=False)
        rows.append(dict(train_fraction=frac, n_train=len(d_sub["ytr"]), mode=mode, test_acc=r["acc"], macro_f1=r["f1_macro"]))
transfer_df = pd.DataFrame(rows)
display(transfer_df.pivot(index="train_fraction", columns="mode", values="macro_f1").round(3))

plt.figure(figsize=(7, 4))
sns.lineplot(data=transfer_df, x="train_fraction", y="macro_f1", hue="mode", marker="o")
plt.title("Transfer learning vs. amount of Swahili training data (" + ("English speech" if EN_REAL else "SYNTHETIC source, pipeline check only") + ")"); plt.xscale("log"); plt.tight_layout(); plt.show()


# %% [markdown]
# **How to interpret.** If *fine-tune* beats *scratch* mostly at 10–25 % of the data and the gap closes at 100 %, transfer learning mainly helps when Swahili data is scarce. If *frozen* is good, the English-learned features are language-independent. If transfer does **not** help, say why (English and Swahili phoneme inventories differ, the English data is too small or too different, the features are already easy for this small task). A negative result with a sound explanation is still a good result.


# %% [markdown]
# ## Step 11 · Extension: speech-to-text + sentiment pipeline
#
# So far we classify a *fixed* vocabulary. A more flexible system is a **cascade**:
#
# ```
# audio  ──► ASR model (speech → Swahili text) ──► text classifier (text → sentiment)
# ```
#
# * **ASR (automatic speech recognition):** we use OpenAI **Whisper** (multilingual; it supports Swahili). Larger Whisper models (`medium`, `large`) are much more accurate for Swahili than `small`. A Swahili-fine-tuned Wav2Vec2-CTC model is an alternative.
# * **Text classifier:** a multilingual XLM-RoBERTa sentiment model (`cardiffnlp/twitter-xlm-roberta-base-sentiment`). It was fine-tuned on multilingual tweets; Swahili is not among its fine-tuning languages, but XLM-R was pre-trained on Swahili text so it often transfers. For better quality, fine-tune it on an AfriSenti Swahili subset (this connects to Group 3).
# * **Error propagation:** a transcription mistake changes the text and may flip the sentiment. We therefore measure **WER** (word error rate) if reference transcripts are available.


# %%
RUN_EXTENSION = True   # needs internet to download the models

if RUN_EXTENSION:
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=dev)
    sentiment = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", device=dev)

    def speech_to_sentiment(audio, sr):
        # audio: 1-D float array, sr: its sample rate
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        text = asr({"raw": audio, "sampling_rate": 16000},
                   generate_kwargs={"language": "swahili", "task": "transcribe"})["text"].strip()
        s = sentiment(text)[0] if text else {"label": "n/a", "score": 0.0}
        return text, s["label"], round(float(s["score"]), 3)

    # 1) Test the text stage on its own with a few Swahili sentences
    for t in ["Napenda sana huduma hii, ni nzuri sana!", "Huduma ni mbaya sana, sijafurahi.", "Kesho nitaenda sokoni."]:
        print(t, "->", sentiment(t)[0])


# %%
if RUN_EXTENSION and CFG.data_source != "synthetic":
    # 2) Full pipeline on a few real test clips (ground-truth word vs. ASR transcript)
    rows = []
    for w, y_true in list(zip(data["wav_te"], data["y_wav_te"]))[:15]:
        text, lab, score = speech_to_sentiment(w, CFG.sr)
        rows.append(dict(true_word=data["classes"][y_true], asr_text=text, sentiment=lab, confidence=score))
    pipe_df = pd.DataFrame(rows); display(pipe_df)

    # 3) Word error rate: how often does ASR output differ from the true word?
    import jiwer
    wer = jiwer.wer(list(pipe_df["true_word"].str.lower()),
                    list(pipe_df["asr_text"].str.lower().str.replace(r"[^\w\s]", "", regex=True)))
    print(f"WER on this small sample: {wer:.2%}")
else:
    print("Skipped the audio demo (synthetic data is not speech). With the real dataset this cell transcribes test clips and scores their sentiment.")


# %% [markdown]
# **Things to discuss:** (1) ASR errors vs. classifier errors – which dominates? (2) Isolated words carry little sentiment, so evaluate on *sentences* (e.g. Common Voice) for a meaningful test; (3) compare the cascade with a direct speech-emotion model (an end-to-end alternative that skips text).


# %% [markdown]
# ## Step 12 · Report guide and checklist
#
# The assignment asks for this notebook **plus a short report**. A good structure, with where each item comes from in this notebook:
#
# 1. **Dataset overview** – source, number of clips/classes, sample rates, duration, label distribution (*Step 3 plots*), any filtering you did (`max_classes`, `min_per_class`).
# 2. **Preprocessing** – resample, trim, fixed length, MFCC+delta, normalisation, augmentation (*Step 5*), train/val/test split and how you avoided leakage.
# 3. **Model architecture** – the 1D CNN diagram and parameter count (*Step 6*), training set-up (loss, AdamW, dropout, scheduler, early stopping) (*Step 7*).
# 4. **Results** – training curves, accuracy, macro-F1, confusion matrix, embeddings (*Step 8*); experiment tables and bar chart (*Step 9*); Wav2Vec2 comparison and transfer-learning curves (*Step 10*).
# 5. **Insights and robustness** – which words are confused and why; effect of sampling rate and resolution; effect of augmentation on generalisation; transfer learning in low-data settings; limitations (single seed, speaker leakage, small data).
# 6. **Extension** – ASR → sentiment pipeline, example outputs, WER, error propagation (*Step 11*).
#
# **Final checklist**
# - [ ] `data_source` switched from `"synthetic"` to the real dataset and the notebook re-run top to bottom
# - [ ] Split done by speaker if speaker ids exist
# - [ ] Test set used only once for final numbers
# - [ ] All plots have titles and labelled axes
# - [ ] Notebook saved with outputs (*File → Save a copy*), plus the report as PDF/Word


# %%
# Save the trained baseline model and a results summary for the report
os.makedirs("outputs", exist_ok=True)
torch.save({"state_dict": model.state_dict(), "classes": data["classes"], "cfg": CFG.__dict__}, "outputs/cnn1d_baseline.pt")
all_res.to_csv("outputs/experiment_results.csv", index=False)
for _name in ["seeds_df", "noise_df", "transfer_df"]:
    if _name in globals():
        globals()[_name].to_csv(f"outputs/{_name}.csv", index=False)
print("Saved: outputs/cnn1d_baseline.pt, outputs/experiment_results.csv")
