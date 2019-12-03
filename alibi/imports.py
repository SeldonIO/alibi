import logging

logger = logging.getLogger(__name__)

_ALTAIR_PRESENT = False
_MATPLOTLIB_PRESENT = False

try:
    import altair  # noqa

    _ALTAIR_PRESENT = True
except ImportError:
    logger.info("altair not found")

try:
    import matplotlib.pyplot as plt  # noqa

    _MATPLOTLIB_PRESENT = True
except ImportError:
    logger.info("matplotlib not found")

_BACKENDS_PRESENT = [_ALTAIR_PRESENT, _MATPLOTLIB_PRESENT]
