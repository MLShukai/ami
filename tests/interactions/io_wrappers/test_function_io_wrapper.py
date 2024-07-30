from pytest_mock import MockerFixture

from ami.interactions.io_wrappers.function_wrapper import FunctionIOWrapper


class TestFunctionIOWrapper:
    def test_wrap(self, mocker: MockerFixture):
        wrap_function = mocker.Mock(return_value="wrapped")
        wrapper = FunctionIOWrapper(wrap_function)
        assert wrapper.wrap("input") == "wrapped"
