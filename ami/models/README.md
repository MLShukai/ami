# Modelsについて

## 深層モデルの実装方法

推論スレッド上で直接的に用いられる最上層のモデルクラスは [`BaseModel`クラス](./base_model.py)を継承する。そして `models/`モジュールの直下に配置する。すなわち、次のように実装される。

```py
from .base_model import BaseModel

class ObservationEncoder(BaseModel):
    ...
```

それ以外の内部的にコンポーネントとして用いられるクラス(Residual Blockなど)は `models/components/`の下に配置する。
