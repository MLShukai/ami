# プロジェクト構造について

## 概要

この資料では、AMIのモジュール、クラス、名前空間といったもののうち、骨子に関わる抽象クラス、重要実装クラスに関してその関係性等を記述する。

## AMI

### `threads` モジュール内部構造

```mermaid
classDiagram
    direction TB

    class ThreadTypes {<<enumeration>>}
    note for ThreadTypes "threadsモジュール内の至る所から参照される。"

    class BaseThread {
        <<abstract>>
    }
    BaseThread --> SharedObjectsPool: Member
    BaseThread --> SharedObjectsFromThisThread: Member
    SharedObjectsPool --> SharedObjectsFromThisThread: Item

    class SharedObjectsPool{
    }

    class SharedObjectsFromThisThread {
    }

    BaseThread <|.. MainThread: Implements
    class MainThread {

    }

    MainThread --> WebApiHandler: Member
    MainThread --> ThreadController: Member
    WebApiHandler --> ThreadController: Member

    BaseThread <|-- BackgroundThread: Inheritance
    class BackgroundThread {
    }

    BackgroundThread <|.. InferneceThread: Implements
    class InferneceThread {
    }
    InferneceThread --> ThreadCommandHandler: Member

    BackgroundThread <|.. TrainingThread: Implements
    class TrainingThread {
    }
    TrainingThread --> ThreadCommandHandler: Member

    namespace thread_control {
        class ThreadController{
        }

        class ThreadCommandHandler {
        }
    }
    ThreadController <--> ThreadCommandHandler: Cross Ref
```

### Inference - Training Relation

mermaidによるクラス図が非常に読みにくいため [draw.ioで作成した図](https://drive.google.com/file/d/1Iggc4oiVy6N04svYzsXPzqratjnqwSVc/view?usp=sharing)を最初に記載する。

![inference-training-relations](./InferenceTrainingRelations.drawio.png)

```mermaid
classDiagram
    direction TB

    namespace threads{
        class InferenceThread
        class TrainingThread
    }

    namespace interactions {

        class Interaction
        class Environment
        class Agent
    }

    namespace models {
        class ModelWrapper
        class InferenceWrapper
        class ModelWrappers["ModelWrappers(Dict)"]
        class InferenceWrappers["InferenceWrappers(Dict)"]
    }

    namespace data {
        class DataBuffer
        class DataCollector
        class DataUser
        class DataCollectors["DataCollectors(Dict)"]
        class DataUsers["DataUsers(Dict)"]
    }

    namespace trainers {
        class Trainers["Trainers(List)"]
        class Trainer
    }

   InferenceThread --> InferenceWrappers: Member
    InferenceThread --> DataCollectors: Member

    InferenceThread --> Interaction: Member
    Interaction --> Environment: Member
    Interaction --> Agent: Member
    Agent --> InferenceWrapper: Member
    Agent --> DataCollector: Member

    TrainingThread --> ModelWrappers: Member
    TrainingThread --> DataUsers: Member

    ModelWrappers <--> InferenceWrappers: Cross Ref
    ModelWrappers --> ModelWrapper: Item
    InferenceWrappers --> InferenceWrapper: Item
    ModelWrapper <--> InferenceWrapper: Cross Ref

    DataCollectors <--> DataUsers: Cross Ref
    DataCollectors --> DataCollector: Item
    DataUsers --> DataUser: Item
    DataCollector <--> DataUser: Cross Ref
    DataCollector --> DataBuffer: Member
    DataUser --> DataBuffer: Member

    TrainingThread --> Trainers: Member
    Trainers --> Trainer: Item
    Trainer --> ModelWrapper: Member
    Trainer --> DataUser: Member
```
