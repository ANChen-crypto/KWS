# `test_streaming` 使用說明

`test_streaming` 會將一段錄音切成重疊音訊視窗，逐窗執行 CTC 關鍵字辨識，並把預測類別畫在整段音訊的時間軸上。這個功能用來觀察觸發與類別變化，不會計算命中率。

## 執行方式

在專案根目錄執行：

```powershell
.\.venv\Scripts\python.exe main.py --test-streaming --wav-path ".\data\recording.wav"
```

使用指定的 checkpoint：

```powershell
.\.venv\Scripts\python.exe main.py --test-streaming --wav-path ".\data\recording.wav" --ckpt ".\weights\conformer_ctc.pth"
```

調整 CTC beam-search 分數門檻：

```powershell
.\.venv\Scripts\python.exe main.py --test-streaming --wav-path ".\data\recording.wav" --streaming-threshold -6
```

也可以同時指定 checkpoint 和門檻：

```powershell
.\.venv\Scripts\python.exe main.py --test-streaming --wav-path ".\data\recording.wav" --ckpt ".\weights\conformer_ctc.pth" --streaming-threshold -5.5
```

`--wav-path` 必須指定要分析的 WAV 檔案路徑，可以是相對於專案根目錄的路徑或絕對路徑。`--ckpt` 未指定時，會使用 `training/train_cfg.yaml` 中 `input_type` 對應的預設權重。`--streaming-threshold` 預設為 `-6.0`；視窗的 beam-search 分數低於此值，或預測序列無法對應到已知關鍵字時，會歸到 `noise` 類別。此分數是解碼分數，不是百分比或機率。

## 輸入資料與視窗設定

WAV 路徑會由 `--wav-path` 傳給 `CTCMelSpecNoisyNpzDataset`。音訊會重採樣到資料設定中的取樣率（目前為 16 kHz）。

目前的串流視窗設定為：

- 視窗長度：2 秒
- 滑動步長：0.2 秒
- 每個視窗獨立執行模型與 CTC beam search

因為第一個完整視窗需要累積 2 秒音訊，圖表開頭的 0–2 秒會標示為 `noise`。之後每個預測會對應到該視窗結束後的時間區間；錄音尾端不足一個步長的區段會截在實際音訊長度。

## 圖表與輸出

執行時會顯示時間軸圖，並將 PNG 存到專案根目錄的 `streaming_timeline` 資料夾。資料夾會在執行時自動建立。檔名包含使用的門檻與 checkpoint 的 `best_val_loss`，例如：

```text
streaming_timeline/streaming_timeline-th_-6.0-id_0.0500.png
```

圖表以時間（秒）為 X 軸、預測類別為 Y 軸，顏色對應如下：

| 類別 | 顏色 |
|---|---|
| help | 藍色 |
| fall | 橘色 |
| need | 綠色 |
| no need | 紅色 |
| noise／未觸發 | 淺灰色 |

`data/dataset.py` 的 noisy dataset 也會將各個 2 秒輸入視窗另存為 `hsiu_noisy_saved/<視窗索引>.wav`，供檢查輸入音訊使用。

## 常見問題

- **找不到輸入 WAV：** 確認 `--wav-path` 指向存在且可讀取的 WAV 檔案。
- **找不到 checkpoint：** 確認預設權重檔存在，或透過 `--ckpt` 指定有效的 checkpoint。
- **預測太常出現或太少出現：** 調整 `--streaming-threshold`。提高門檻（例如從 `-6` 調到 `-5`）會使較多視窗被歸為 `noise`；降低門檻則會接受更多預測。
