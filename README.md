# ニューラルネットワーク

## 概要
MNISTを読み込み，数字の認識を行う．
- 入力層: 28x28の画素値
- 隠れ層: 128の隠れ層
- 出力層: 0~9の数値のone hot表現

隠れ層ではReLU関数，出力層ではSoftmax関数を用いた．

## 実装方法
- バックプロパゲーションを用いた．
- 単純な設計にした．
  - 型はほぼ`double`と`int`のみで構成されている．
  - ヘッダファイルは標準的なものだけを使用している．

## 実行
Git Bash for WindowsとGNU Make for Windowsを使用した．実行環境に合わせて適宜`Makefile`を書き換えること．

```
$ make
$ cd out
$ ls
error_data.csv test.exe train.exe weight_data.csv

$ ./train.exe
$ ./test.exe
```

## ライセンス
[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png "CC0")](http://creativecommons.org/publicdomain/zero/1.0/deed.ja)

