from enum import Enum

import pytest

from alibi.utils import _get_options_string


class TestEnum(str, Enum):
    OPTION_A = "option_a"
    OPTION_B = "option_b"


@pytest.mark.parametrize("test_enum, expected_str",
                         [(TestEnum, "'option_a' | 'option_b'"), ])
def test__get_options_string(test_enum, expected_str):
    assert _get_options_string(test_enum) == expected_str
