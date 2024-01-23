# AMI: Autonomous Machine Intelligence

## 開発環境セットアップ

### Docker

VRChatやOBSなどのホストOSに依存したものを除いた、Pythonなどの開発環境はDockerイメージにまとめてある。
事前に次のツールをインストールしておく。

なお、Linux系OS上から実行することを前提とする。

- docker
- make

```sh
make docker-build
make docker-run
```
