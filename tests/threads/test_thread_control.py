"""このテストコードは実際のシナリオを模擬したテストロジックを用いている。

このテストプログラムがハングアップし終了しない場合も異常状態であることに注意。
"""

import copy
import threading
import time

import pytest
from pytest_mock import MockerFixture

from ami.threads.thread_control import (
    Checkpointing,
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
    assert not handler.is_loop_paused()

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
    assert handler.is_loop_paused()

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
    assert not handler.is_loop_paused()

    # 一時停止中でも終了命令を処理できるか。
    controller.pause()
    time.sleep(0.05)
    controller.shutdown()

    thread.join()

    # Shutdownの際に`resume`(復帰）が呼ばれているか
    assert pause_resume_event_log.num_paused == 2
    assert pause_resume_event_log.num_resumed == 2
    assert not handler.is_loop_paused()


def test_create_handlers():
    controller = ThreadController()
    handlers = controller.handlers
    assert ThreadTypes.MAIN not in handlers
    assert ThreadTypes.TRAINING in handlers
    assert ThreadTypes.INFERENCE in handlers


def test_wait_for_loop_pause():
    controller = ThreadController()
    handler = ThreadCommandHandler(controller)

    pause_timer = threading.Timer(0.1, handler._loop_pause_event.set)

    start_time = time.perf_counter()
    pause_timer.start()
    assert handler.wait_for_loop_pause()
    assert handler.is_loop_paused()
    assert time.perf_counter() - start_time >= 0.1

    handler._loop_pause_event.clear()
    assert not handler.wait_for_loop_pause(0.001)
    assert not handler.is_loop_paused()


def test_save_checkpoint(tmp_path, mocker: MockerFixture) -> None:
    controller = ThreadController()
    checkpointing = Checkpointing(tmp_path)
    controller.checkpointing = checkpointing
    mock_save_checkpoint = mocker.Mock()
    ckpt_path = tmp_path / "test.ckpt"
    mock_save_checkpoint.return_value = ckpt_path
    controller.checkpointing.save_checkpoint = mock_save_checkpoint

    for hdlr in controller.handlers.values():
        threading.Timer(0.1, hdlr._loop_pause_event.set).start()

    assert controller.save_checkpoint() == ckpt_path
    mock_save_checkpoint.assert_called_once()

    with pytest.raises(RuntimeError):
        controller = ThreadController(0.001)
        controller.save_checkpoint()
