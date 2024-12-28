# Ref:
# https://github.com/jik876/hifi-gan/blob/master/models.py


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)
