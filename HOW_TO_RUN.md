# How to run everything yourself

Use the normal `python` (Python 3.13, has the GPU build of torch). Do NOT activate `.venv` (CPU-only torch).

## 0. Open a terminal in the project
Open PowerShell, then:
```powershell
cd C:\Users\arsh\.gemini\antigravity\scratch\turbine_blade_defect_mvp
```
Check the GPU works (should print `True`):
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## 1. Predict one image (terminal), about 10 s
```powershell
python inference/predict_single.py --image "data/figshare/WT blade defect dataset/JPEGImages/4.jpg"
```
Expected: crack, ~89.5%, severity 5.
Changed input: replace `4.jpg` with `754.jpg` (wrong: craze crack, 46%), `10.jpg`, `3.jpg`, or any number 0 to 1064.
Note: the "Action Protocol" text it prints is a fixed rule per class/level, not learned. Say so.

## 2. Demo page (browser)
```powershell
python -m streamlit run app/demo_app.py
```
Opens http://localhost:8501. Sidebar: demo images, any test image (with a "Random test image" button for
"try another input"), any image, or upload. "Show annotated boxes" draws the dataset's HUMAN boxes in orange;
the model itself never draws boxes, it only outputs type + severity. "Under the hood" tabs show all 6
probabilities, the 4 ordinal questions, and the pipeline/limitations.
If port 8501 is busy, an old copy is still running: close that terminal window first.

Dispatch board (top switch "Dispatch board (simulation)"): "Dispatch <image>" sends the current image's
result to a repair-team queue; "Dispatch 5 random" adds 5 test images; "Reset" clears it. Queues are ordered
by severity, then arrival. Confidence < 60% goes to the Human review box, which forwards to teams (dotted lines).
SAY: "This is a simulation. Routing and ordering are fixed rules on the model output; nothing here is learned."
Only show it if asked how the system would be used. It is not part of the 8 minutes.
Code: app/dispatch_board.py.
Stop it: click the terminal and press Ctrl+C.
Shortcut for 1 + 2: double-click `RUN_DEMO.bat`.

## 3. Re-score the saved models (proves the slide numbers), about 1 min
```powershell
python scripts/verify_test_metrics.py
```
Expected: A 93.71%, B 94.34%, C 92.45%, McNemar p = 0.69, list of the 12 wrong images.
```powershell
python scripts/severity_baseline.py
```
Expected: MAE 0.981 / 0.535 / 0.415 / 0.346.

## 4. Training (do NOT run live, takes ~5 min per model): just show the command
```powershell
python training/multitask_trainer.py --seed 42
```
Settings live in `config_unified.yaml`.

## Code to open if asked
| They ask about | Open |
|---|---|
| Model / CBAM / heads | `models/blade_multitask_net.py` (CBAM class line 66, heads ~line 175) |
| Loss (focal, ordinal, Smooth L1) | `training/multitask_losses.py` |
| Training loop, optimizer, scheduler | `training/multitask_trainer.py` |
| Leak-free split, severity rule | `scripts/make_group_splits.py` (BASE_SEVERITY) |
| Augmentation | `datasets/augmentations.py` |
| All numbers | `RESULTS_VERIFIED.md` |
