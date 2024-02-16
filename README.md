# AMI: Autonomous Machine Intelligence

## 開発環境セットアップ

### VRChatとの連携

[vrchat-ioのドキュメンテーション](https://github.com/Geson-anko/vrchat-io?tab=readme-ov-file#vrchat)を参考に、VRChatやOBSをインストールする。

NVIDIA Driver の **バージョン 525以下**ではSteamが正常に動作しないため注意してください。(提供されている最新の安定版Driverを使うこと。)

### Docker

VRChatやOBSなどのホストOSに依存したものを除いた、Pythonなどの開発環境はDockerイメージにまとめてある。
事前に次のツールをインストールしておく。

なお、Linux OS上から実行することを前提とする。

- docker

  Dockerの公式ドキュメンテーションを参考にインストールを行う。
  <https://docs.docker.com/engine/install/>

  NVIDIA Container Toolkitをインストールする。
  <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

  NOTE: [**インストールしただけで満足せず、コンテナラインタイムをDocker用にConfigureすること**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)

- make

  Unix系 OSではプリインストールされていることが多い。

次のコマンドでイメージをビルドし、起動する。Dockerを実行する際`sudo`が必要になるが、[現在ログインしているユーザーをDockerグループに所属させることで回避することができる。](https://qiita.com/DQNEO/items/da5df074c48b012152ee)

```sh
# project root. (ami/)
make docker-build
make docker-run
```

後はVSCodeなどのお好みのエディタからDockerコンテナにアタッチし、`/workspace`ディレクトリで作業を行う。このディレクトリは永続ボリュームであるため、万が一 Dockerのコンテナインスタンスを削除しても作業内容は保存される。
