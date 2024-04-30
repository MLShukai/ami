import pytest

from ami.threads.utils import ThreadSafeFlag


class TestThreadSafeFlag:
    @pytest.fixture
    def flag(self) -> ThreadSafeFlag:
        return ThreadSafeFlag()

    def test_is_set_when_inital(self, flag):
        assert flag.is_set() is False, "Flag should be initially False"

    def test_set(self, flag):
        flag.set()
        assert flag.is_set() is True, "Flag should be set to True"

    def test_clear(self, flag):
        flag.set()
        flag.clear()
        assert flag.is_set() is False, "Flag should be cleared to False"

    def test_take(self, flag):
        assert flag.take() is False
        flag.set()
        assert flag.take() is True, "Get should return True"
        assert flag.is_set() is False, "Flag should be cleared after get"
