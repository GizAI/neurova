from __future__ import annotations


DEFAULT_SYSTEM = (
    "You are Neurova, a concise local assistant. Answer the user's request directly. "
    "Use one or two clear sentences unless the user asks for a list. If the answer is unknown, say you do not know."
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASSISTANT_HEADER = f"{IM_START}assistant\n"


def format_chatml_user_prompt(user: str, system: str = DEFAULT_SYSTEM) -> str:
    return (
        f"{IM_START}system\n{system.strip()}\n{IM_END}\n"
        f"{IM_START}user\n{user.strip()}\n{IM_END}\n"
        f"{ASSISTANT_HEADER}"
    )


def format_chatml_pair(user: str, assistant: str, system: str = DEFAULT_SYSTEM) -> str:
    return f"{format_chatml_user_prompt(user, system)}{assistant.strip()}\n{IM_END}"


def strip_chatml_tail(text: str) -> str:
    text = text.split(IM_END, 1)[0]
    text = text.split(IM_START, 1)[0]
    return text.strip()
