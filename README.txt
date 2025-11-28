推論程式使用說明

==============================================
環境準備
==============================================
1. 建立虛擬環境：
   conda create -n dl python=3.12

2. 進入環境：
   conda activate dl

3. 安裝 Python 套件：
   pip install -r requirements.txt

4. 安裝 FFmpeg：
   Windows: 下載 https://ffmpeg.org/download.html
   macOS: brew install ffmpeg
   Linux: sudo apt-get install ffmpeg

==============================================
執行推論
==============================================

1. 確認檔案結構：
   ├── artist20/
   │   └── test/              # 放置測試音樂檔案
   ├── checkpoint_dl.pth     # 訓練好的模型檔案
   └── DL_inference.py       # 推論程式

2. 準備測試資料：
   - 將要分類的 MP3 音樂檔案放在 artist20/test/ 資料夾
   - 檔案命名建議：001.mp3, 002.mp3, 003.mp3, ...

3. 執行推論：
   python DL_inference.py

4. 查看結果：
   - 預測結果會儲存在 test_pred_dl.json 檔案中
   - 格式範例：
     {
       "001": ["artist1", "artist2", "artist3"],
       "002": ["artist4", "artist5", "artist6"]
     }

5. 根據測試需求修改預測結果檔案名稱
