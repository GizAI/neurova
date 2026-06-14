from __future__ import annotations


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def chatml(system: str, user: str, assistant: str | None = None) -> str:
    text = (
        f"{IM_START}system\n{system.strip()}{IM_END}\n"
        f"{IM_START}user\n{user.strip()}{IM_END}\n"
        f"{IM_START}assistant\n"
    )
    if assistant is not None:
        text += f"{assistant.strip()}{IM_END}"
    return text


def assistant_start_marker() -> str:
    return f"{IM_START}assistant\n"
