# Modelsについて

全てのモデルは一度 `ModelWrapper`クラスにラップされた状態でAMI上では実行される。

エージェントの内部で用いられる、推論上の最上位クラス(ObservationEncoder, ForwardDynamicsなど）は `./models`の直下に配置する。

推論時だけ処理を変えたい場合は、 `ModelWrapper`を継承し、`infer`メソッドをオーバーライドする。この際に入力テンソルを適切な演算デバイスに送る必要がある。`ModelWrapper.device`属性で取得できるが、内部パラメータを直接参照しているため取得コストが高い可能性がある。

## コンポーネントに関して

内部的にコンポーネントとして用いられるクラス(Residual Blockなど)は `models/components/`の下に配置する。
