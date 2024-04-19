"""このテストコードは実際のシナリオを模擬したテストロジックを用いている。

このテストプログラムがハングアップし終了しない場合も異常状態であることに注意。
"""

import copy
import threading
import time

from ami.threads.thread_control import (
    ThreadCommandHandler,
    ThreadController,
    ThreadTypes,
)


class Counter:
    def __init__(self) -> None:
        self._v = 0
        self._lock = threading.RLock()

    @property
    def value(self) -> int:
        with self._lock:
            return copy.copy(self._v)

    def increment(self) -> None:
        with self._lock:
            self._v += 1


class PauseResumeEventLog:
    def __init__(self) -> None:
        self.num_paused = 0
        self.num_resumed = 0

    def on_paused(self) -> None:
        self.num_paused += 1

    def on_resumed(self) -> None:
        self.num_resumed += 1


def infinity_increment_thread(counter: Counter, handler: ThreadCommandHandler) -> None:
    """テスト用のバックグラウンドスレッド。カウンタを無限に増加。"""
    while handler.manage_loop():
        counter.increment()
        time.sleep(0.001)


def test_shutdown() -> None:
    """Shutdown Flagが正常に機能しているかをチェック."""
    controller = ThreadController()
    handler = ThreadCommandHandler(controller)
    counter = Counter()
    init_value = counter.value

    thread = threading.Thread(target=infinity_increment_thread, args=(counter, handler))
    thread.start()

    time.sleep(0.05)

    controller.shutdown()
    assert controller.is_shutdown() is True

    thread.join()

    assert init_value < counter.value


def test_manage_loop() -> None:
    controller = ThreadController()
    handler = ThreadCommandHandler(controller, check_resume_interval=10)
    counter = Counter()
    pause_resume_event_log = PauseResumeEventLog()  # pause/resumeのイベント呼び出し記録用
    handler.on_paused = pause_resume_event_log.on_paused
    handler.on_resumed = pause_resume_event_log.on_resumed

    thread = threading.Thread(target=infinity_increment_thread, args=(counter, handler))
    thread.start()

    # Backgroundスレッドの一時停止が期待通り行われているか
    # 一時停止した -> カウンタが停止している。
    # また、`pause`イベントコールバックが呼び出された。
    controller.pause()
    time.sleep(0.01)
    value = counter.value
    time.sleep(0.05)
    not_changed_value = counter.value

    assert value == not_changed_value
    assert pause_resume_event_log.num_paused == 1

    # Backgroundスレッドの再開処理が期待通り行われているか
    # 再開した -> カウンタが増加している
    # また、`resume`イベントコールバックが呼び出された
    controller.resume()
    time.sleep(0.01)
    value = counter.value
    time.sleep(0.05)
    changed_value = counter.value

    assert value < changed_value
    assert pause_resume_event_log.num_resumed == 1

    # 一時停止中でも終了命令を処理できるか。
    controller.pause()
    time.sleep(0.05)
    controller.shutdown()

    thread.join()

    # Shutdownの際に`resume`(復帰）が呼ばれているか
    assert pause_resume_event_log.num_paused == 2
    assert pause_resume_event_log.num_resumed == 2


def test_create_handlers():
    controller = ThreadController()
    handlers = controller.handlers
    assert ThreadTypes.MAIN not in handlers
    assert ThreadTypes.TRAINING in handlers
    assert ThreadTypes.INFERENCE in handlers
