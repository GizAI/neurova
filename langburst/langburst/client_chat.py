from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import select
import signal
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8008/v1"
DEFAULT_MODEL = "langburst-qwen3.6-27b-q3"
_PROMPT_INTERRUPT = "\x03"
_COMMANDS = ("/exit", "/quit", "/new", "/reset", "/history", "/help", "/thinking")
_THINKING_LEVELS = ("none", "low", "medium", "high")


try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
except Exception:  # pragma: no cover - dependency fallback for broken envs.
    Console = None
    Panel = None
    Text = None

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.events import AppFocus
    from textual.widgets import RichLog, TextArea
except Exception:  # pragma: no cover - plain fallback when textual is absent.
    App = None
    ComposeResult = Any
    Binding = None
    Vertical = None
    AppFocus = None
    RichLog = None
    TextArea = None


class ChatRenderer:
    def __init__(self) -> None:
        self.console = Console(highlight=False) if Console is not None else None

    def header(self, *, base_url: str, model: str, thinking: str) -> None:
        line = f"LangBurst chat: {base_url} model={model} mode=stateless history thinking={thinking}"
        if self.console is None or not sys.stdout.isatty():
            print(line)
            return
        self.console.print(Panel(line, title="LangBurst", border_style="cyan", expand=False))

    def help(self) -> None:
        self.info("Commands: /exit, /quit, /reset, /history, /thinking [none|low|medium|high], /help")

    def info(self, text: str) -> None:
        if self.console is None:
            print(text)
        else:
            self.console.print(text, style="dim", markup=False, highlight=False)

    def user_message(self, text: str) -> None:
        if self.console is None or not sys.stdout.isatty():
            return
        self.console.print(Text("you", style="bold green") + Text(f"  {text}", style="white"))

    def assistant_prefix(self) -> None:
        if self.console is None:
            print("assistant> ", end="", flush=True)
        else:
            self.console.print("assistant> ", style="bold cyan", end="")

    def assistant_delta(self, text: str) -> None:
        if self.console is None:
            print(text, end="", flush=True)
        else:
            self.console.print(text, end="", markup=False, highlight=False)

    def line(self) -> None:
        print()

    def metrics(self, usage: dict[str, Any] | None) -> None:
        self.info(_format_usage_summary(usage))

    def error(self, text: str) -> None:
        if self.console is None:
            print(text, file=sys.stderr)
        else:
            self.console.print(text, style="bold red", markup=False, highlight=False, file=sys.stderr)


class TurnCancelled(Exception):
    pass


class ExitChat(Exception):
    pass


@dataclass
class ChatSession:
    system: str | None = None
    reasoning_effort: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system}] if self.system else []

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": assistant_text})

    @property
    def user_turns(self) -> int:
        return sum(1 for message in self.messages if message["role"] == "user")


def _load_conversation(path: Path, *, system: str | None, reasoning_effort: str | None) -> ChatSession:
    session = ChatSession(system=system, reasoning_effort=reasoning_effort)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return session
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        return session
    clean: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "user", "assistant"} and isinstance(content, str):
            clean.append({"role": role, "content": content})
    if clean:
        session.messages = clean
    saved_effort = payload.get("reasoning_effort") if isinstance(payload, dict) else None
    if reasoning_effort is None and saved_effort in {"none", "low", "medium", "high"}:
        session.reasoning_effort = saved_effort
    return session


def _save_conversation(path: Path, session: ChatSession) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reasoning_effort": session.reasoning_effort,
        "messages": session.messages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class CommandResult:
    handled: bool
    exit: bool = False
    message: str | None = None


def _base_url(value: str) -> str:
    text = value.rstrip("/")
    if text.endswith("/v1"):
        return text
    return f"{text}/v1"


def _json_request(method: str, url: str, payload: dict[str, Any] | None = None, *, timeout: float = 30.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot connect to {url}: {exc.reason}") from exc
    return json.loads(raw) if raw else {}


def _stream_request(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    cancel_event: threading.Event | None = None,
) -> Iterable[dict[str, Any]]:
    req = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    try:
        response = urlopen(req, timeout=timeout)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot connect to {url}: {exc.reason}") from exc

    with response:
        event_lines: list[str] = []
        for raw_line in response:
            if cancel_event is not None and cancel_event.is_set():
                break
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if event_lines:
                    text = "\n".join(event_lines)
                    event_lines.clear()
                    if text == "[DONE]":
                        break
                    yield json.loads(text)
                continue
            if line.startswith("data:"):
                event_lines.append(line[5:].lstrip())


def _load_history_fallback(path: Path) -> None:
    try:
        import readline
    except Exception:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            readline.read_history_file(path)
    except OSError:
        return


def _save_history_fallback(path: Path) -> None:
    try:
        import readline
    except Exception:
        return
    try:
        readline.write_history_file(path)
    except OSError:
        return


def _load_input_history(path: Path, *, limit: int = 500) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    history: list[str] = []
    for line in lines:
        text = line.rstrip("\n")
        if text and (not history or history[-1] != text):
            history.append(text)
    return history[-limit:]


def _save_input_history(path: Path, history: list[str], *, limit: int = 500) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean: list[str] = []
    for item in history:
        text = item.strip()
        if not text:
            continue
        if not clean or clean[-1] != text:
            clean.append(text)
    path.write_text("\n".join(clean[-limit:]) + ("\n" if clean else ""), encoding="utf-8")


def _prompt_fallback(prefix: str) -> str | None:
    try:
        return input(prefix)
    except EOFError:
        return None
    except KeyboardInterrupt:
        print()
        return _PROMPT_INTERRUPT


def _thinking_status(reasoning_effort: str | None) -> str:
    if reasoning_effort == "none":
        return "none"
    if reasoning_effort:
        return reasoning_effort
    return "default"


def _set_thinking(arg: str, *, reasoning_effort: str | None) -> tuple[str | None, str]:
    value = arg.strip().lower()
    if value in {"", "status"}:
        return reasoning_effort, f"thinking: {_thinking_status(reasoning_effort)}"
    if value in {"none", "low", "medium", "high"}:
        return value, f"thinking: {value}"
    raise ValueError("usage: /thinking [none|low|medium|high]")


def _handle_command(text: str, session: ChatSession) -> CommandResult:
    if not text.startswith("/"):
        return CommandResult(handled=False)
    if text in {"/exit", "/quit"}:
        return CommandResult(handled=True, exit=True)
    if text == "/help":
        return CommandResult(handled=True, message="Commands: /exit, /quit, /new, /reset, /history, /thinking [none|low|medium|high], /help")
    if text == "/history":
        return CommandResult(handled=True, message=f"turns={session.user_turns}")
    if text in {"/new", "/reset"}:
        session.reset()
        return CommandResult(handled=True, message="new conversation")
    if text.startswith("/thinking"):
        try:
            session.reasoning_effort, message = _set_thinking(
                text[len("/thinking") :],
                reasoning_effort=session.reasoning_effort,
            )
        except ValueError as exc:
            return CommandResult(handled=True, message=str(exc))
        return CommandResult(handled=True, message=message)
    return CommandResult(handled=True, message=f"unknown command: {text}")


def _chat_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    text: str,
    max_tokens: int | None,
    temperature: float,
    top_p: float | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    request_messages = [*messages, {"role": "user", "content": text}]
    payload: dict[str, Any] = {
        "model": model,
        "messages": request_messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    return payload


def _format_metric(value: Any, *, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _format_usage_summary(usage: dict[str, Any] | None) -> str:
    if not usage:
        return "metrics> usage unavailable"
    perf = usage.get("performance") or {}
    parts = [
        f"prefill={_format_metric(perf.get('prefill_tok_s'), suffix=' tok/s')}",
        f"decode={_format_metric(perf.get('decode_tok_s'), suffix=' tok/s')}",
    ]
    return "metrics> " + " ".join(parts)


def _stream_turn(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    cancel_event: threading.Event | None,
    on_delta: Callable[[str], None],
) -> tuple[str, dict[str, Any] | None]:
    full_text = ""
    usage: dict[str, Any] | None = None
    for event in _stream_request(
        f"{base_url}/chat/completions",
        payload,
        timeout=timeout,
        cancel_event=cancel_event,
    ):
        if cancel_event is not None and cancel_event.is_set():
            break
        if "error" in event:
            message = event["error"].get("message", event["error"])
            raise RuntimeError(str(message))
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        text = delta.get("content") or ""
        if text:
            on_delta(text)
            full_text += text
    return full_text, usage


@contextlib.contextmanager
def _turn_cancel_controls():
    cancel_event = threading.Event()
    exit_event = threading.Event()
    previous_sigint = signal.getsignal(signal.SIGINT)
    old_termios = None
    stop_reader = threading.Event()

    def on_sigint(_signum, _frame):
        if cancel_event.is_set():
            exit_event.set()
        cancel_event.set()
        raise TurnCancelled()

    def esc_reader() -> None:
        if not sys.stdin.isatty():
            return
        while not stop_reader.is_set() and not cancel_event.is_set():
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if readable:
                    ch = os.read(sys.stdin.fileno(), 1)
                    if ch == b"\x1b":
                        cancel_event.set()
                        try:
                            import _thread

                            _thread.interrupt_main()
                        except Exception:
                            pass
                        return
            except Exception:
                return

    try:
        signal.signal(signal.SIGINT, on_sigint)
        if sys.stdin.isatty():
            old_termios = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        thread = threading.Thread(target=esc_reader, name="langburst-cli-cancel", daemon=True)
        thread.start()
        yield cancel_event, exit_event
    finally:
        stop_reader.set()
        signal.signal(signal.SIGINT, previous_sigint)
        if old_termios is not None:
            with contextlib.suppress(Exception):
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_termios)


def _run_turn(base_url: str, payload: dict[str, Any], *, timeout: float) -> tuple[str, dict[str, Any] | None, bool]:
    full_text = ""
    usage: dict[str, Any] | None = None
    cancelled = False
    with _turn_cancel_controls() as (cancel_event, exit_event):
        try:
            full_text, usage = _stream_turn(
                base_url,
                payload,
                timeout=timeout,
                cancel_event=cancel_event,
                on_delta=lambda text: print(text, end="", flush=True),
            )
        except TurnCancelled:
            cancelled = True
            if exit_event.is_set():
                raise ExitChat()
        except KeyboardInterrupt:
            cancelled = True
    print()
    print(_format_usage_summary(usage))
    if cancelled:
        print("cancelled")
    return full_text, usage, cancelled


if TextArea is not None and Binding is not None:

    class ChatOutput(RichLog):
        can_focus = False


    class ChatComposer(TextArea):
        BINDINGS = [
            Binding("enter", "submit", "Send", priority=True),
            Binding("alt+enter", "newline", "Newline", priority=True),
            Binding("ctrl+j", "newline", "Newline", priority=True, show=False),
            Binding("up", "history_previous", "Previous input", priority=True, show=False),
            Binding("down", "history_next", "Next input", priority=True, show=False),
        ]

        def action_submit(self) -> None:
            self.app.action_send()

        def action_newline(self) -> None:
            self.insert("\n")

        def action_history_previous(self) -> None:
            if not self.app.composer_history_previous():
                self.action_cursor_up()

        def action_history_next(self) -> None:
            if not self.app.composer_history_next():
                self.action_cursor_down()

else:
    ChatOutput = RichLog
    ChatComposer = TextArea


class LangBurstTextualApp(App if App is not None else object):
    AUTO_FOCUS = "#composer"
    CSS = """
    Screen {
        background: $surface;
    }
    #root {
        height: 100%;
    }
    #output {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
        scrollbar-gutter: stable;
    }
    #composer {
        height: 7;
        border: round $secondary;
    }
    """
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+l", "focus_input", "Focus input"),
        ("ctrl+r", "reset", "Reset"),
        ("ctrl+q", "quit", "Quit"),
        ("f1", "help", "Help"),
    ]

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        max_tokens: int | None,
        temperature: float,
        top_p: float | None,
        timeout: float,
        system: str | None,
        reasoning_effort: str | None,
        conversation_file: Path,
        history_file: Path,
    ) -> None:
        super().__init__()
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.conversation_file = conversation_file
        self.history_file = history_file
        self.session_state = _load_conversation(conversation_file, system=system, reasoning_effort=reasoning_effort)
        self.input_history = _load_input_history(history_file)
        self.history_index: int | None = None
        self.history_draft = ""
        self.cancel_event: threading.Event | None = None
        self.transcript_text = ""
        self.current_answer = ""
        self.busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            yield ChatOutput(id="output", wrap=True, markup=False, highlight=False)
            yield ChatComposer("", id="composer")

    def on_mount(self) -> None:
        self.title = "LangBurst"
        self.sub_title = self.model
        self._append_log(f"LangBurst: {self.base_url} model={self.model}")
        self._append_log("Enter send | Alt+Enter newline | Esc cancel | Ctrl+L focus input | /new clears saved chat")
        self._restore_transcript_from_session()
        self._set_status("ready")
        self._focus_composer()

    def _log(self):
        return self.query_one("#output", ChatOutput)

    def _composer(self):
        return self.query_one("#composer", ChatComposer)

    def _focus_composer(self) -> None:
        with contextlib.suppress(Exception):
            self.set_focus(self._composer(), scroll_visible=False)

    def on_app_focus(self, _event: AppFocus) -> None:
        if not self.busy:
            self._focus_composer()

    def _render_transcript(self) -> None:
        log = self._log()
        log.clear()
        if self.transcript_text:
            log.write(self.transcript_text)
        log.scroll_end(animate=False)

    def _append_log(self, text: str = "") -> None:
        if self.transcript_text:
            self.transcript_text += "\n"
        self.transcript_text += text
        self._render_transcript()

    def _restore_transcript_from_session(self) -> None:
        for message in self.session_state.messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                continue
            self._append_log(f"{role}> {content}")

    def _save_session(self) -> None:
        with contextlib.suppress(OSError):
            _save_conversation(self.conversation_file, self.session_state)

    def _record_input_history(self, text: str) -> None:
        clean = text.strip()
        if not clean or clean.startswith("/"):
            return
        if self.input_history and self.input_history[-1] == clean:
            self.history_index = None
            self.history_draft = ""
            return
        self.input_history.append(clean)
        self.input_history = self.input_history[-500:]
        self.history_index = None
        self.history_draft = ""
        with contextlib.suppress(OSError):
            _save_input_history(self.history_file, self.input_history)

    def _replace_composer_text(self, text: str) -> None:
        composer = self._composer()
        composer.load_text(text)
        with contextlib.suppress(Exception):
            lines = text.splitlines() or [""]
            composer.move_cursor((len(lines) - 1, len(lines[-1])))

    def composer_history_previous(self) -> bool:
        composer = self._composer()
        if not self.input_history or not composer.cursor_at_first_line:
            return False
        if self.history_index is None:
            self.history_draft = composer.text
            self.history_index = len(self.input_history) - 1
        else:
            self.history_index = max(0, self.history_index - 1)
        self._replace_composer_text(self.input_history[self.history_index])
        return True

    def composer_history_next(self) -> bool:
        composer = self._composer()
        if self.history_index is None or not composer.cursor_at_last_line:
            return False
        if self.history_index >= len(self.input_history) - 1:
            self.history_index = None
            self._replace_composer_text(self.history_draft)
            self.history_draft = ""
        else:
            self.history_index += 1
            self._replace_composer_text(self.input_history[self.history_index])
        return True

    def _set_status(self, text: str) -> None:
        thinking = _thinking_status(self.session_state.reasoning_effort)
        self.sub_title = f"{self.model} | {text} | thinking={thinking} | turns={self.session_state.user_turns}"

    def action_focus_input(self) -> None:
        self._focus_composer()

    def action_help(self) -> None:
        self._append_log("Commands: /exit, /quit, /new, /reset, /history, /thinking [none|low|medium|high], /help")

    def action_reset(self) -> None:
        self.session_state.reset()
        self.transcript_text = ""
        self._render_transcript()
        self._append_log("reset")
        self._save_session()
        self._set_status("ready")

    def action_cancel(self) -> None:
        if self.cancel_event is not None:
            self.cancel_event.set()
            self._set_status("cancelling")
        else:
            self._focus_composer()

    def action_send(self) -> None:
        if self.busy:
            self._set_status("busy")
            return
        text = self._composer().text.strip()
        if not text:
            return
        self._composer().load_text("")
        self.history_index = None
        self.history_draft = ""
        result = _handle_command(text, self.session_state)
        if result.handled:
            if text in {"/new", "/reset"}:
                self.transcript_text = ""
                self._render_transcript()
                self._append_log(result.message or "new conversation")
                self._save_session()
            elif result.message:
                self._append_log(result.message)
            self._set_status("ready")
            if result.exit:
                self.exit()
            return
        self._record_input_history(text)
        self._append_log(f"you> {text}")
        self.current_answer = ""
        self._append_log("assistant> ")
        self.busy = True
        self.cancel_event = threading.Event()
        self._set_status("running")
        self.run_worker(lambda: self._generate(text), thread=True, exclusive=False)

    def _generate(self, text: str) -> None:
        payload = _chat_payload(
            model=self.model,
            messages=self.session_state.messages,
            text=text,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            reasoning_effort=self.session_state.reasoning_effort,
        )
        usage: dict[str, Any] | None = None
        cancelled = False
        try:
            answer, usage = _stream_turn(
                self.base_url,
                payload,
                timeout=self.timeout,
                cancel_event=self.cancel_event,
                on_delta=lambda delta: self.call_from_thread(self._append_delta, delta),
            )
            cancelled = bool(self.cancel_event and self.cancel_event.is_set())
        except Exception as exc:
            self.call_from_thread(self._finish_error, str(exc))
            return
        self.call_from_thread(self._finish_turn, text, answer, usage, cancelled)

    def _append_delta(self, delta: str) -> None:
        self.current_answer += delta
        self.transcript_text += delta
        self._render_transcript()

    def _finish_turn(self, user_text: str, answer: str, usage: dict[str, Any] | None, cancelled: bool) -> None:
        if answer:
            self.session_state.add_turn(user_text, answer)
            self._save_session()
        self._append_log(_format_usage_summary(usage))
        if cancelled:
            self._append_log("cancelled")
        self.current_answer = ""
        self.cancel_event = None
        self.busy = False
        self._set_status("ready")
        self._focus_composer()

    def _finish_error(self, message: str) -> None:
        self._append_log(f"error: {message}")
        self.current_answer = ""
        self.cancel_event = None
        self.busy = False
        self._set_status("error")
        self._focus_composer()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive LangBurst API chat client")
    parser.add_argument("--base-url", default=os.environ.get("LANGBURST_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("LANGBURST_MODEL", DEFAULT_MODEL))
    cli_max_tokens = os.environ.get("LANGBURST_CLI_MAX_TOKENS")
    parser.add_argument("--max-tokens", type=int, default=int(cli_max_tokens) if cli_max_tokens else None)
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("LANGBURST_CLI_TEMPERATURE", "0")))
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("LANGBURST_CLI_TIMEOUT_S", "600")))
    parser.add_argument("--system", default=None)
    parser.add_argument("--history-file", type=Path, default=Path(os.environ.get("LANGBURST_CLI_HISTORY", "~/.langburst_chat_history")).expanduser())
    parser.add_argument(
        "--conversation-file",
        type=Path,
        default=Path(os.environ.get("LANGBURST_CLI_CONVERSATION", "~/.langburst_chat_conversation.json")).expanduser(),
    )
    parser.add_argument("--thinking-level", choices=("none", "low", "medium", "high"), default=None, help="set reasoning_effort")
    parser.add_argument("--plain", action="store_true", help="use the non-TUI fallback client")
    args = parser.parse_args()

    base_url = _base_url(args.base_url)
    if not args.plain and sys.stdin.isatty() and sys.stdout.isatty() and App is not None:
        LangBurstTextualApp(
            base_url=base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            system=args.system,
            reasoning_effort=args.thinking_level,
            conversation_file=args.conversation_file,
            history_file=args.history_file,
        ).run()
        return

    if sys.stdin.isatty() and sys.stdout.isatty():
        _load_history_fallback(args.history_file)
    session = _load_conversation(args.conversation_file, system=args.system, reasoning_effort=args.thinking_level)
    renderer = ChatRenderer()

    renderer.header(base_url=base_url, model=args.model, thinking=_thinking_status(session.reasoning_effort))
    renderer.help()
    last_interrupt_s: float | None = None

    try:
        while True:
            text = _prompt_fallback("you> ")
            if text is None:
                break
            if text == _PROMPT_INTERRUPT:
                now = time.monotonic()
                if last_interrupt_s is not None and now - last_interrupt_s <= 2.0:
                    break
                last_interrupt_s = now
                print("press Ctrl-C again to exit")
                continue
            text = text.strip()
            if not text:
                continue
            last_interrupt_s = None
            result = _handle_command(text, session)
            if result.handled:
                if result.message:
                    renderer.info(result.message)
                if text in {"/new", "/reset"}:
                    _save_conversation(args.conversation_file, session)
                if result.exit:
                    break
                continue

            payload = _chat_payload(
                model=args.model,
                messages=session.messages,
                text=text,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                reasoning_effort=session.reasoning_effort,
            )
            renderer.assistant_prefix()
            try:
                answer, _usage, cancelled = _run_turn(base_url, payload, timeout=args.timeout)
            except ExitChat:
                print("\nexit")
                break
            except RuntimeError as exc:
                print(f"\nerror: {exc}", file=sys.stderr)
                continue
            if cancelled:
                last_interrupt_s = time.monotonic()
                continue
            session.add_turn(text, answer)
            _save_conversation(args.conversation_file, session)
    finally:
        _save_history_fallback(args.history_file)


if __name__ == "__main__":
    main()
