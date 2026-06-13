from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ByteTokenizer:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    byte_offset: int = 3

    @property
    def vocab_size(self) -> int:
        return 259

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(byte + self.byte_offset for byte in text.encode("utf-8", errors="replace"))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytearray()
        for token_id in ids:
            if self.byte_offset <= int(token_id) < self.byte_offset + 256:
                raw.append(int(token_id) - self.byte_offset)
        return raw.decode("utf-8", errors="replace")
