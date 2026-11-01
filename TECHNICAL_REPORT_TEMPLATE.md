# Keyword Spotting (KWS) Technical Report

> **Instructions:** Replace every `[placeholder]` with project-specific information. Remove sections that do not apply, and label unmeasured results as **Not measured** rather than estimating them.

## Document control

| Field | Value |
|---|---|
| Report title | [Project / experiment name] |
| Version | [Version] |
| Date | [YYYY-MM-DD] |
| Author(s) | [Name(s)] |
| Reviewer(s) | [Name(s)] |
| Model / checkpoint | CNN CTC (`input_type: cnn_ctc`); configured default: `./weights/cnn_ctc.pth` (checkpoint presence/version not verified) |
| Code revision | [Git commit or release tag] |
| Status | [Draft / Reviewed / Approved] |

## 1. Executive summary

This report documents the repository's configured CNN-CTC keyword-spotting training path. It predicts phoneme sequences for selected Taiwanese/Hokkien phrases from 35-bin log-Mel NPZ inputs, using a CTC objective and an auxiliary binary cross-entropy objective. The configured training run is 50 epochs with batch size 32 and AdamW at learning rate 0.0001. Dataset files are read from hard-coded external `G:` paths. FAR, FRR, latency, and deployment performance are **Not measured in this report**.

### Key results

| Measure | Result | Evaluation set / conditions |
|---|---:|---|
| False accept rate (FAR) | Not measured | Must report protocol and denominator |
| False reject rate (FRR) | Not measured | Must report protocol and denominator |
| Accuracy / macro F1 | Not measured | No evaluation run recorded here |
| Detection latency | Not measured | Target device not specified |
| Model size / peak memory | Not measured | Target artifact/device not specified |

## 2. Scope and requirements

### Objective

The configured CTC NPZ dataset trains phoneme sequences for selected Taiwanese/Hokkien and English phrase labels, plus unknown examples. Intended users and deployment environment are not specified in the inspected files.

### Supported vocabulary

| Class / phrase | Label or ID | Language / pronunciation | Positive examples |
|---|---|---|---:|
| 救命喔 | `1_1` | `k iu m ia o h` | Not available (external NPZ data) |
| 我跌倒了 | `1_2` | `g ua p uah t o a` | Not available (external NPZ data) |
| 需要 | `1_3` | `s u iau` | Not available (external NPZ data) |
| 毋免 (mapped to the no-need evaluation class) | `1_6` | `m b ian` | Not available (external NPZ data) |
| Unknown / out-of-vocabulary | `9_9`, `unk` | Single phoneme `unk` | Not available (external NPZ data) |

### Operating requirements

| Requirement | Target | Verification method |
|---|---|---|
| FAR | Not specified | No target documented |
| FRR | Not specified | No target documented |
| Detection latency | Not specified | No target documented |
| Supported sampling rates / devices | Dataset config specifies 16 kHz; target devices not specified | No device test matrix documented |
| Compute, memory, and power limits | Not specified | No target measurement documented |
| Privacy / retention constraints | Not specified | Requires dataset-owner confirmation |

## 3. System overview

The training path consumes precomputed 35-bin log-Mel arrays from NPZ files, predicts frame-level phoneme log probabilities and an auxiliary binary output, then applies CTC beam decoding for evaluation. This training path does not document microphone capture or a downstream action.

```text
NPZ log-Mel features -> CNN -> phoneme log probabilities -> CTC beam decoder
                     -> exact phoneme-sequence mapping -> keyword / unknown
```

### Components

| Component | Implementation / version | Input -> output |
|---|---|---|
| Audio capture | Not specified in this training path | Not specified |
| Preprocessing | NPZ loader reads `mel` or `mels`, first 35 bins; train-only masking/gain augmentation | Float feature tensor |
| Feature extraction | Saved log-Mel input; dataset config specifies 16 kHz, 35 Mel bins, FFT 512, hop 160 | `(batch, 35, time)`; expected time length 197 |
| Acoustic model | 10-block residual 1D CNN with 64 feature channels | Frame logits `(time, batch, 37)` and binary logits `(batch, 2)` |
| Decoder / trigger logic | Greedy-width CTC prefix beam search and exact phoneme-sequence mapping during evaluation | Keyword class or unknown |
| Runtime / target | PyTorch; training selects CUDA when available, otherwise CPU | N657B0 |

## 4. Data and labeling

### Data sources

| Split / source | Speakers | Utterances | Duration | Collection conditions |
|---|---:|---:|---:|---|
| Training | Not available | Not available | Not available | `G:/npzs_stride_0.01s_train` plus eligible folders under `G:/testing_set_wav/npz_o` |
| Validation / evaluation loader | Not available | Not available | Not available | In `train()` the held-out loader is used for per-epoch evaluation; `test_ctc()` also evaluates its configured test loader |
| Test | Not available | Not available | Not available | NPZ files under selected folders ending in `person_2_xinyi` or `an_xinyi` in `G:/testing_set_wav/npz_o` |
| Background / noise | Not available | Not available | Not available | Active CTC NPZ loader uses `9_9` and generated `unk` labels; explicit noise inclusion is not established |

### Split and label protocol

The active `cnn_ctc` route loads `CTCMelSpecNpzDataset`. It selects labels `1_1`, `1_2`, `1_3`, `1_6`, and `9_9`, then appends samples from separate `npz_unk_train` / `npz_unk_test` roots. Labels are phoneme ID sequences; CTC uses blank ID 0. Evaluation maps exact decoded sequences to five broad classes, with unknown as class 4. The `class_map_13` table and `num_classes: 13` setting do not define the CNN-CTC output vocabulary in this route. Train and evaluation data draw from hard-coded external paths; verify speaker/session separation and absence of leakage before treating results as held-out performance.

### Data quality and governance

Not documented in the inspected code/configuration. Dataset counts, consent/provenance, and retention policy require confirmation from the dataset owner.

### Augmentation

For training only, NPZ features receive a random frequency mask of 0–5 Mel bins and a random time mask of 0–15 frames; masks are filled with zero. Features whose path contains `testing_set_wav` additionally receive gain scaling in the range 0.5–1.1, applied as an additive dB offset. No augmentation is applied by this code path to evaluation samples.

## 5. Signal processing

| Setting | Value |
|---|---|
| Input encoding / channels | Precomputed NPZ feature array; source audio encoding/channel count not available here |
| Sampling rate | 16,000 Hz in `data/dataset.yaml` |
| Resampling method | Not established for the saved NPZ features |
| Window duration and hop | Dataset config declares 32,000 samples (2 s), but the NPZ CTC loader does not apply that audio length directly; feature hop is 160 samples (10 ms at 16 kHz) |
| Normalization / gain control | Training feature gain augmentation on selected NPZ paths; other normalization not established |
| Feature type and parameters | Log-Mel; 35 bins, FFT 512, hop 160; NPZ loader reads first 35 rows |
| Silence / clipping handling | Not established for this NPZ training path |

The NPZ dataset supplies fixed-shape features (time masking assumes 197 frames). The CNN downsamples time by three stride-2 convolutions and crops two frames from each end, yielding 21 output frames for the expected input. Streaming window and live microphone preprocessing are outside this training path.

## 6. Model and decision logic

### Architecture

`CNN(input_dim=35)` uses three stride-2 Conv1d subsampling layers, then 10 residual blocks. Each block has two 1D convolutions (64 -> 128 -> 64 channels), BatchNorm, ReLU, and dropout 0.2. The CTC head emits 37 phoneme/blank log-probability channels; the auxiliary head emits two logits from a 21-frame representation. There is no pretrained/frozen component in this CNN definition. Parameter count depends on the instantiated model and has not been recorded here.

### Training objective

| Item | Value |
|---|---|
| Prediction target | Framewise phoneme sequence (CTC; blank ID 0) plus binary auxiliary target |
| Loss function(s) and weights | `CTCLoss(blank=0, zero_infinity=True, reduction="none")` plus unweighted `CrossEntropyLoss` on binary logits. Per-sample CTC losses are normalized by their batch mean, clamped to 0.5–2.0, multiplied by sample `importance` (3.0 for paths containing `testing_set_wav`, otherwise 1.0), averaged, and added to binary CE. Binary target is 1 when target length is 1, otherwise 0. `loss_weight: [1,1,1,1,1]` is not used by the `cnn_ctc` criterion. |
| Optimizer and learning rate schedule | AdamW, learning rate 0.0004, weight decay 0.01; no scheduler configured in the inspected trainer |
| Batch size and epochs | Batch size 32; 50 epochs |
| Initialization / pretrained weights | No pretrained initialization shown; training may optionally load a supplied checkpoint and optimizer state |
| Checkpoint selection criterion | Best held-out loss is tracked; checkpoint default path is `./weights/cnn_ctc.pth`. Exact checkpoint availability and training-run artifact were not verified. |
| Random seeds | Not configured in the inspected files |

### Decoding and trigger policy

Evaluation uses `beam_search_decode`; its implementation forces beam size 1 (greedy-width prefix search) and defaults to emission threshold `ln(0.01)`. Decoded phoneme sequences are mapped by exact match; unmatched sequences become unknown. `test_ctc()` sweeps decoder scores from -6 to 0 in 151 steps, accepting a prediction when its score meets the threshold and otherwise assigning unknown. These are decoder scores, not calibrated probabilities. Temporal smoothing, debounce/cooldown, and live trigger reset behavior are not specified here.

## 7. Experimental setup

The active configuration and code path are identified below. Dataset NPZs reside on external `G:` paths, and no run logs, dependency lockfile, hardware record, or checkpoint checksum were verified; a full reproduction therefore needs those artifacts and environment details.

| Item | Details |
|---|---|
| Code revision | Not recorded |
| Configuration | `training/train_cfg.yaml`: `input_type=cnn_ctc`, `num_classes=13`, `epoch=50`, `batch_size=32`, `learning_rate=0.0004`, `loss_weight=[1,1,1,1,1]` |
| Checkpoint | Default `./weights/cnn_ctc.pth`; existence and checksum not verified |
| Runtime and dependencies | PyTorch; exact package versions not recorded |
| Hardware | CUDA if available, otherwise CPU; exact hardware not recorded |
| Evaluation command / procedure | `python main.py --test-ctc [--ckpt <checkpoint>]`; procedure sweeps CTC score thresholds in `test_ctc()` |
| Test-set integrity | Test folders are selected by suffix under `G:/testing_set_wav/npz_o`; training also reads other folders under that root. Confirm speaker/session isolation externally. |

## 8. Evaluation methodology

### Metrics and definitions

The evaluation code defines FAR/FRR over labeled NPZ samples, not over audio hours or continuous streams. It does not define an event-matching time tolerance.

- **FAR:** False alarms among unknown samples divided by the number of unknown samples; reported as a fraction/percentage, not per hour.
- **FRR:** Positive samples whose predicted class differs from the reference divided by the number of positive samples. This includes both missed/unknown predictions and wrong-keyword confusions.
- **Accuracy / precision / recall / F1:** The criterion checks exact phoneme-sequence equality during evaluation; `test_ctc()` also creates a 5-class confusion matrix. A summary accuracy/F1 aggregation is not defined in that method.
- **Latency:** Not measured; no timing protocol is implemented in this evaluation path.
- **Resource use:** Not measured.

### Threshold selection

`test_ctc()` sweeps 151 score thresholds linearly from -6 to 0 and computes FAR/FRR at each threshold. The inspected code does not select or lock a final operating threshold, nor establish that threshold selection is separate from the final test set.

### Evaluation conditions

Evaluation conditions beyond the NPZ test-folder selection are not documented. Microphone, distance, noise level, speaker count, overlap, and continuous-stream conditions are not available in the inspected configuration/code.

## 9. Results

### Overall results

| Model / setting | Threshold | FAR | FRR | Accuracy / F1 | Latency | Notes |
|---|---:|---:|---:|---:|---:|---|
| CNN CTC | Sweep -6 to 0 (151 points) | Not measured | Not measured | Not measured | Not measured | Requires checkpoint and external NPZ data to run |

### Per-class results

| Class | Support | Precision | Recall | F1 | False triggers / confusions |
|---|---:|---:|---:|---:|---|
| help / 救命喔 | Not available | Not measured | Not measured | Not measured | No evaluation outputs supplied |
| fall / 我跌倒了 | Not available | Not measured | Not measured | Not measured | No evaluation outputs supplied |
| need / 需要 | Not available | Not measured | Not measured | Not measured | No evaluation outputs supplied |
| no_need / 毋免 | Not available | Not measured | Not measured | Not measured | No evaluation outputs supplied |
| unk | Not available | Not measured | Not measured | Not measured | No evaluation outputs supplied |

### Error analysis

No evaluation outputs or representative error cases were available to analyze. Re-run evaluation with the intended checkpoint and NPZ test data, then record threshold-specific confusions and test conditions.

## 10. Deployment and runtime

The CNN-CTC training code does not establish a deployed model, target integration, or conversion parity. Record these only after validating the exported artifact against the training checkpoint.

| Measure | Float model | Deployed model | Device / method |
|---|---:|---:|---|
| Artifact size | Not measured | Not measured | No artifact measurements supplied |
| Peak memory | Not measured | Not measured | No device measurements supplied |
| Inference time | Not measured | Not measured | No latency measurements supplied |
| FAR / FRR after conversion | Not measured | Not measured | No deployed-model evaluation supplied |

### Operational behavior

Not documented for a deployed runtime.

## 11. Limitations, risks, and mitigations

| Limitation / risk | Impact | Mitigation or next step |
|---|---|---|
| Dataset is loaded from hard-coded external `G:` paths and counts are unavailable | Results cannot be reproduced from checked-in files alone | Record dataset manifest, counts, provenance, and checksums |
| Train and evaluation datasets use different folder selections under a shared external root | Speaker/session leakage cannot be ruled out from repository code alone | Audit split membership and document speaker-level separation |
| Configured `loss_weight` is ignored by the selected `cnn_ctc` criterion | Changing this config value will not reweight the CTC/binary losses | Document this behavior or change the criterion/config deliberately |
| No measured operating threshold, FAR/FRR result, or deployment benchmark is included | Requirements and deployment readiness cannot be concluded | Run threshold evaluation on a verified held-out set and target device |

## 12. Conclusions and next steps

Requirements cannot yet be assessed: no thresholds are specified and no measured test or deployment results were supplied. Before reporting performance, verify split isolation, preserve the data/checkpoint/configuration identifiers, run `test_ctc()` on the intended held-out set, select an operating threshold on validation data, and measure latency/resources on the target device.

| Priority | Action | Owner | Target date |
|---|---|---|---|
| High | Verify speaker/session split isolation and record a dataset manifest | TBD | TBD |
| High | Run the threshold sweep on a verified held-out dataset and record results | TBD | TBD |
| Medium | Select a validation operating threshold and evaluate the deployed model on target hardware | TBD | TBD |

## Appendix A. Reproduction checklist

- [ ] Training, validation, and test splits are identified and reproducible.
- [ ] Preprocessing and label mappings are documented.
- [ ] Configuration, code revision, checkpoint, and random seeds are recorded.
- [ ] Thresholds were selected without using the final test set.
- [ ] FAR / FRR definitions and evaluation conditions are explicit.
- [ ] Exported or quantized model was evaluated on the target runtime.
- [ ] Plots, logs, and result files are identified and archived.

## Appendix B. References and artifacts

- Configuration: [`training/train_cfg.yaml`](./training/train_cfg.yaml)
- Dataset labels and NPZ loading: [`data/dataset.py`](./data/dataset.py)
- CNN architecture: [`modeling/cnn.py`](./modeling/cnn.py)
- CTC and auxiliary loss implementation: [`criterion/criterion.py`](./criterion/criterion.py)
- Trainer, data split selection, and FAR/FRR sweep: [`training/trainer.py`](./training/trainer.py)
- External dataset manifest, checkpoint, evaluation plots/logs, and target-device measurements: Not supplied
