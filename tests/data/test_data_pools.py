from ami.data.data_pools import StepData


class TestStepData:
    def test_copy(self):
        sd = StepData()
        sd["a"] = [1, 2, 3]

        copied = sd.copy()
        assert sd is not copied
        assert sd["a"] is not copied["a"]
