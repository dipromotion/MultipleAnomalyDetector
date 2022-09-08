# 概要
以下、動画中の、外観検査AI装置に複数不良の同時検出と処理の高速化機能を追加するプログラムです。

![外観検査AI 装置の作り方②_mini](https://user-images.githubusercontent.com/106806108/189013637-8abc51a7-8721-4f1a-b3c0-d250850dad31.png)

# 事前準備
Python及び以下のライブラリのインストールが必要です。

- 必要なライブラリ
1. OpenCV
2. requests
3. ONNX
- ライブラリインストールコマンド

```
pip install opencv-python
pip install requests
pip install onnx
```
```
動作確認済みバージョン（参考）
python==3.8.12
opencv==4.0.1
requests==2.27.1
onnx==1.10.2
```


# 使い方

CustomVisionポータルより作成したAIモデルをONNX形式でエクスポートして、modelフォルダの中にすべてのファイルを格納して実行してください。

モデルは変数modelとして以下のように読み込まれます。
```
model = Model("model/model.onnx")
```

- 取得方法は動画にて説明しておりますのでそちらをご参照ください。

# 作成者
株式会社神戸デジタル・ラボ kdl-saiki

連絡先： [https://www.kdl.co.jp/contact/](https://www.kdl.co.jp/contact/)

# ライセンス
MIT
