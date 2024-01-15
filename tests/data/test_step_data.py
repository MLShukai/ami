from ami.data.step_data import StepData


class TestStepData:
    def test_copy(self) -> None:
        sd = StepData()
        sd["a"] = [1, 2, 3]

        copied = sd.copy()
        assert sd is not copied
        assert sd["a"] is not copied["a"]
