"""독립 난수 스트림.

기존 프로젝트에서 배운 것: 공유 ``rng`` 하나에서 순서대로 뽑으면 앞쪽 로직을
한 줄 고치는 순간 뒤따르는 모든 엔티티의 잠재값이 밀린다. 스크랩 배치 하나를
바꿨더니 데이터셋 전체가 달라졌던 그 문제다.

그래서 엔티티마다 ``(seed, 용도, 키)``로 시드를 유도해 독립 스트림을 만든다.
보고 생성 로직을 고쳐도 공장 잠재값은 그대로다.
"""

from __future__ import annotations

import hashlib

import numpy as np


def stream(seed: int, purpose: str, key: str = "") -> np.random.Generator:
    h = hashlib.blake2b(f"{seed}|{purpose}|{key}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big"))
