# data

## data bufferについて

新しい経験バッファは [`buffers/`以下の`BaseDataBuffer`クラス](./buffers/base_data_buffer.py)を継承して作成する。作成するクラスは新たに `buffers`下に`*_buffer.py`を作成し、そこに記述する。ファイル名はスネークケース、クラス名はアッパーキャメルケースを採用し、名前は基本一致させること。
