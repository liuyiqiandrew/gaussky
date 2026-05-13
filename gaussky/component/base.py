from typing import Protocol
from dataclasses import dataclass


class GaussianComponent(Protocol):
    name: str

    def sample_map(
        self,
        *,
        nside,
        lmax
    ):
        ...