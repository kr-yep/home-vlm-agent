"""Force model output into Taiwan Traditional Chinese.

VLMs drift into Simplified characters even when asked not to, so every string
shown to the user goes through OpenCC's s2twp (Simplified to Taiwan, with
Taiwanese phrasing) instead of trusting the prompt.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _converter():
    from opencc import OpenCC

    return OpenCC("s2twp")


def to_taiwan(text: str) -> str:
    return _converter().convert(text) if text else text
