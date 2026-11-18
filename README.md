# Group 0 – Swahili speech (word) classification

Course: CSC 63106 Artificial Intelligence · Week 6 practical · Common task for all groups

A 1D CNN that listens to a short clip of one spoken Swahili word and predicts which word it is.
The project also includes sampling-rate / spectrogram / architecture experiments, a noise-robustness test,
English → Swahili transfer learning, Wav2Vec2 fine-tuning, and a speech-to-text → sentiment extension.

## What is in this folder

| File | What it is |
|---|---|
| `Group0_Swahili_Speech_Classification.ipynb` | The main notebook (explanations + code + outputs). **Start here.** |
| `Group0_Swahili_Speech_Classification.py` | The same code as a script. Every `# %%` line is a cell (works in VS Code, Spyder, or plain `python`). |
| `requirements.txt` | Python packages. |
| `Group0_Swahili_Speech_Classification_Report.docx` | The short report. |
| `data/` *(you download it)* | Swahili words dataset (parquet shards, about 3.9 GB). |
| `data_en/` *(you download it, optional but needed for the transfer experiment)* | English Speech Commands (parquet). |

The notebook and the script contain identical code. Use whichever you prefer.

---

## 1. Before you run anything

### Python version
Use **Python 3.10, 3.11 or 3.12**. Very new Python versions (for example 3.14) may not have working builds of
PyTorch / librosa / numba yet. If you have several versions on Windows, `py -0` lists them and
`py -3.12 -m venv venv` creates an environment with the right one.

### Hardware
* **CPU only is enough** for Steps 1–9 (the CNN has about 266 thousand parameters; one training run takes a few minutes).
* **A GPU** (Colab T4 is fine) is needed for Step 10a (Wav2Vec2) and recommended for Step 11 (Whisper).
  Without a GPU those cells skip themselves or can be switched off (`RUN_EXTENSION = False`).

### Download the data (once)

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('michsethowusu/swahili-words-speech-text-parallel', repo_type='dataset', local_dir='./data')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('pollen-robotics/speech-commands-v0.02', repo_type='dataset', local_dir='./data_en')"
```

* The first command is the **Swahili words dataset** (411,048 one-word clips, columns `audio` and `text`).
  It is required.
* The second is **English Speech Commands** (one-second clips of single English words, columns `audio` and `label`).
  It is only needed for the English → Swahili transfer experiment (Step 10b).
  **If `./data_en` is missing, Step 10b falls back to synthetic tones and prints a warning. Those numbers must not be reported as English → Swahili transfer.**
  English Common Voice (named in the assignment) contains full sentences, so it does not give clean word labels; Speech Commands is the word-level substitute.
* Keep both folders **next to the notebook** (same folder), because the code looks for `./data` and `./data_en`.

The first time Step 2 runs, it reads only the text column of every shard to find the 10 most frequent words, then
loads a balanced sample (150 clips per word) and **saves it as `swahili_sample_10w_150n_100m_s42.pkl`** (tens of MB).
Next runs load that small file and take seconds. You can copy that file to another computer or to Colab instead of the 3.9 GB dataset.

---

## 2. Run it – pick ONE option

### Option A · Google Colab (easiest if you have no GPU)

1. Go to <https://colab.research.google.com> → **File → Upload notebook** → choose the `.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Get the data into the Colab session. Either
   * **download it there** – add a cell at the top and run it (takes several minutes; the files vanish when the session resets):
     ```python
     !pip -q install huggingface_hub
     from huggingface_hub import snapshot_download
     snapshot_download('michsethowusu/swahili-words-speech-text-parallel', repo_type='dataset', local_dir='./data')
     snapshot_download('pollen-robotics/speech-commands-v0.02', repo_type='dataset', local_dir='./data_en')
     ```
   * **or upload the small `.pkl` sample files** (see the note above) with the folder icon on the left. You then do not need `./data`.
4. Run the cells from the top with **Shift+Enter** (or **Runtime → Run all**).
   The first code cell installs the libraries automatically on Colab.
5. Download your results before the session ends (`outputs/` folder, plus **File → Download .ipynb**).

### Option B · VS Code (or a VS Code based editor such as Kiro)

1. Install the **Python** and **Jupyter** extensions.
2. Open the project folder (`File → Open Folder`).
3. Create the environment and install packages in the VS Code terminal:
   ```powershell
   py -3.12 -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
   (Linux/macOS: `python3.12 -m venv venv && source venv/bin/activate`. With an NVIDIA GPU use the CUDA command from <https://pytorch.org/get-started/locally/> instead of the CPU one.)
   If PowerShell refuses to activate: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
4. Open the `.ipynb`, click **Select Kernel** (top right) → choose the `venv` interpreter, then **Run All** or run cell by cell.
   *Or* open the `.py` file: each `# %%` line shows **Run Cell**, and the results appear in the Interactive Window.

### Option C · Jupyter (PowerShell / cmd / terminal)

```powershell
cd path\to\project
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
jupyter lab
```
A browser tab opens; open the notebook and press **Shift+Enter** on each cell. Keep the terminal window open.

### Option D · Plain Python script (no notebook)

```powershell
python Group0_Swahili_Speech_Classification.py                 # plots open in windows, one after another
$env:HEADLESS="1"; python Group0_Swahili_Speech_Classification.py   # no windows: plots saved to outputs/figures/
```
(bash: `HEADLESS=1 python Group0_Swahili_Speech_Classification.py`.)

---

## 3. What each step produces and roughly how long it takes (CPU, 8 GB RAM)

| Step | What | Time |
|---|---|---|
| 2 | Load audio (first run only: reads all 16 shards, then caches the sample) | a few minutes once, seconds afterwards |
| 3–4 | Metadata and plots | seconds |
| 5 | Preprocessing, augmentation, MFCC | about 20 s |
| 6–8 | Build, train (about 13 epochs), evaluate | a few minutes |
| 9a/9b | 19 short experiments | tens of minutes |
| 9c | 3 seeds × 2 settings | tens of minutes |
| 9d | Noise robustness | a few minutes |
| 10a | Wav2Vec2 fine-tuning | **needs a GPU**; skipped without one |
| 10b | English → Swahili transfer (needs `./data_en`) | tens of minutes on CPU |
| 11 | Whisper + XLM-R sentiment | downloads a few GB; **GPU recommended**; set `RUN_EXTENSION = False` to skip |

Tip: while testing, lower `per_class` (for example 60) and `epochs` in the config cell so a full pass takes minutes.
Put the final run back to `per_class = 150`.

---

## 4. Settings you may want to change (Step 1 config cell)

| Setting | Default | Meaning |
|---|---|---|
| `data_source` | `"parquet"` | `"parquet"` (downloaded dataset), `"folder"` (`root/<word>/*.wav`), `"hf"` (Hugging Face name), `"synthetic"` (fake tones, for a quick test) |
| `parquet_dir` | `"./data"` | Folder with the `*.parquet` shards (searched recursively) |
| `max_classes`, `per_class` | 10, 150 | Number of words and clips per word |
| `sr`, `duration` | 16000, 0.5 | Sample rate and fixed clip length (99 % of trimmed words are 0.45 s or shorter) |
| `n_mfcc`, `n_mels`, `use_delta` | 40, 64, True | Feature size |
| `aug_copies` | 2 | Augmented copies of each training clip (noise, pitch shift, time shift) |
| `n_layers`, `kernel`, `activation`, `dropout` | 3, 5, relu, 0.3 | CNN settings |
| `epochs`, `patience`, `lr` | 15, 4, 1e-3 | Training settings |

---

## 5. Troubleshooting

* **`No .parquet files under './data'`** – the notebook is not in the same folder as `data/`, or the download did not finish. Check with `Get-ChildItem .\data -Recurse -Filter *.parquet` (should list 16 shards).
* **`Column 'text' not found`** – the error message lists the real column names; change `hf_audio_col` / `hf_label_col` in the config.
* **Kernel dies / out of memory** – close other programs, lower `per_class` or `aug_copies`.
* **`pip` is not recognised** – use `python -m pip install ...`.
* **Wrong Python in the notebook** – run `import sys; print(sys.version)` in a cell. If it is not the version of your `venv`, choose the right kernel (Jupyter: *Kernel → Change Kernel*; VS Code: *Select Kernel*), or run `python -m ipykernel install --user --name swahili` inside the activated venv.
* **MP3 files cannot be read** – convert to WAV or install ffmpeg.
* **Step 10a prints "No GPU detected"** – expected on a CPU-only machine; run that step on Colab.
* **Results differ slightly between runs** – seeds are fixed (`seed=42`), but different library versions or hardware can change results in the last decimal. Differences of 1–2 accuracy points between experiments are within noise on this small test set (225 clips); that is why Step 9c repeats the key comparison with several seeds.

## 6. Reproducing the report

Run the notebook top to bottom with the real data, then copy the numbers and figures into the report:
Step 3 (data), Step 8 (baseline results), Step 9 (experiments, seeds, noise), Step 10b (transfer), Step 11 (extension).
Step 12 of the notebook maps each report section to the step that produces it. Results are also saved to `outputs/` (CSV tables, model weights).

## 7. Known limits (state them in the report)

* The dataset card does not list speaker IDs and does not say who recorded the clips, so we **cannot split by speaker**.
  The random split may put the same voice in train and test, which can make accuracy look better than it would for a new speaker.
* We use the 10 most frequent words, sampled to 150 clips each (balanced). The full dataset is not balanced
  (`na` has about 24,000 clips; `ni` about 3,200).
* Test set = 225 clips, so one clip is worth about 0.44 accuracy points.
