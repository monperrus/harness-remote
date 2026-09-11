#!/usr/bin/env python3
"""ACP (Agent Client Protocol) adapter exposing agentknit agents over stdio JSON-RPC.

Implements the surface harness-remote's bridge consumes:
  initialize, authenticate, session/new, session/list, session/resume,
  session/prompt, session/cancel, session/set_config_option

One adapter process serves one agentknit "face" (model + endpoint + client
adaptations), configured via environment variables so harness-remote can be
pointed at it with --acp-command/--acp-arg:

  AGENTKNIT_ACP_MODEL      model id passed to agentknit.load_specification
  AGENTKNIT_ACP_ENDPOINT   /chat/completions endpoint URL (or run://...)
  AGENTKNIT_ACP_FACES      optional JSON array of extra faces for the model
                           catalog: [{"id": "...", "name": "...", "model": "...",
                           "endpoint": "..."}]

Stdlib + agentknit only; no other dependencies.
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, os.environ.get("AGENTKNIT_SRC", "/home/martin/workspace/prototypes/agentknit"))

# agentknit prints informational notices ("Using cached probe …") to stdout,
# which would corrupt the JSON-RPC stream. Redirect all library-level stdout
# writes to stderr for the life of the process; adapter output uses _send().
import contextlib  # noqa: E402

_stdout_real = sys.stdout
sys.stdout = sys.stderr

import agentknit  # noqa: E402


class _StdoutProxy:
    """Route agentknit's prints to stderr while _send() uses the real stdout."""

    def write(self, data: str) -> int:
        return sys.stderr.write(data)

    def flush(self) -> None:
        sys.stderr.flush()

    def __getattr__(self, name: str):
        return getattr(sys.stderr, name)

PROTOCOL_VERSION = 1

MODEL = os.environ.get("AGENTKNIT_ACP_MODEL", "deepseek-v4-flash")
ENDPOINT = os.environ.get("AGENTKNIT_ACP_ENDPOINT", "https://api.deepseek.com/v1")
KEY_NAME = os.environ.get("AGENTKNIT_ACP_KEY_NAME", "deepseek_api_key")

EXTRA_FACES = json.loads(os.environ.get("AGENTKNIT_ACP_FACES", "[]"))
FACES = [{"id": MODEL, "name": MODEL, "model": MODEL, "endpoint": ENDPOINT,
          "key_name": KEY_NAME}] + EXTRA_FACES

LOG_DIR = Path(os.environ.get("AGENTKNIT_ACP_STATE", Path.home() / ".local/state/agentknit-acp"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    with (LOG_DIR / "adapter.log").open("a") as f:
        f.write(msg + "\n")


_write_lock = threading.Lock()


def _send(payload: dict) -> None:
    data = json.dumps(payload)
    with _write_lock:
        _stdout_real.write(data + "\n")
        _stdout_real.flush()


def _respond(req_id, result=None, error=None) -> None:
    msg = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result if result is not None else {}
    _send(msg)


def _notify(session_id: str, update: dict) -> None:
    _send({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": session_id, "update": update},
    })


class FaceSession:
    """One ACP session backed by one agentknit session."""

    def __init__(self, session_id: str, cwd: str, face: dict) -> None:
        self.id = session_id
        self.cwd = cwd
        self.face = face
        self.lock = threading.Lock()
        self.cancel_token: agentknit.CancelToken | None = None
        self.turn_thread: threading.Thread | None = None
        self.title: str | None = None
        self.created = _now_ms()
        self.updated = self.created
        self.message_count = 0
        self._first_chunk_ids: dict[str, str] = {}
        self.schema = self._schema()
        self.session = self._init_agentknit_session()

    def _schema(self) -> dict:
        schema = agentknit.load_specification(self.face["model"], self.face["endpoint"])
        schema["endpoint"] = self.face["endpoint"]
        schema["display_name"] = self.face["name"]
        # Force streaming on: cached probes may say the endpoint does not
        # stream, but without stream=True agentknit emits no content_delta
        # events and the ACP client would see nothing until the turn ends.
        schema.setdefault("provider_api_support", {}).setdefault("streaming", {})["supported"] = True
        # Key source: a file ~/.config/agentknit/<name> (mode 600) beats the
        # desktop keyring, which a headless server does not have.
        key_name = self.face.get("key_name")
        if key_name:
            key_file = Path.home() / ".config/agentknit" / key_name
            if key_file.is_file():
                os.environ[key_name.upper()] = key_file.read_text().strip()
                schema["keyring_service"] = "login2"
                schema["keyring_username"] = key_name
        return schema

    def _init_agentknit_session(self):
        old_cwd = os.getcwd()
        try:
            if self.cwd and Path(self.cwd).is_dir():
                os.chdir(self.cwd)
            return agentknit.init_session(
                self.schema,
                non_interactive=True,
                min_cacheable_tokens=int(os.environ.get("AGENTKNIT_ACP_MIN_CACHEABLE", "1024")),
            )
        finally:
            os.chdir(old_cwd)

    # -- streaming bridge -------------------------------------------------

    def _make_on_event(self):
        sid = self.id

        def on_event(event_type: str, data: dict) -> None:
            try:
                self.updated = _now_ms()
                if event_type == "content_delta":
                    _notify(sid, {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": self._chunk_id("assistant"),
                        "content": {"type": "text", "text": data.get("text", "")},
                    })
                elif event_type == "reasoning_delta":
                    _notify(sid, {
                        "sessionUpdate": "agent_thought_chunk",
                        "messageId": self._chunk_id("thought"),
                        "content": {"type": "text", "text": data.get("text", "")},
                    })
                elif event_type == "tool_call":
                    call_id = "call_" + uuid.uuid4().hex[:16]
                    self._open_calls = getattr(self, "_open_calls", [])
                    self._open_calls.append(call_id)
                    _notify(sid, {
                        "sessionUpdate": "tool_call",
                        "toolCallId": call_id,
                        "title": data.get("name", "tool"),
                        "status": "in_progress",
                        "rawInput": data.get("args", {}),
                        "_meta": {"toolName": data.get("name", "tool")},
                    })
                elif event_type == "tool_result":
                    call_id = None
                    if getattr(self, "_open_calls", None):
                        call_id = self._open_calls.pop(0)
                    if call_id:
                        _notify(sid, {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": call_id,
                            "status": "completed",
                            "rawOutput": str(data.get("result", ""))[:10000],
                        })
                elif event_type == "final_answer":
                    pass
                elif event_type == "error":
                    _notify(sid, {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": self._chunk_id("assistant"),
                        "content": {"type": "text",
                                    "text": f"\n\n[error] {data.get('text', 'unknown error')}"},
                    })
            except Exception:
                _log("on_event error:\n" + traceback.format_exc())

        return on_event

    def _chunk_id(self, kind: str) -> str:
        if kind not in self._first_chunk_ids:
            self._first_chunk_ids[kind] = uuid.uuid4().hex
        return self._first_chunk_ids[kind]

    # -- turn lifecycle ----------------------------------------------------

    def prompt(self, text: str) -> str:
        """Run one agentknit turn synchronously (called from a worker thread)."""
        with self.lock:
            self.cancel_token = agentknit.CancelToken()
            self._first_chunk_ids = {}
            self._open_calls = []
            self.session["on_event"] = self._make_on_event()
            self.session["_event_handlers"] = {}
            client = agentknit.create_client(self.schema)
            stop_reason = "end_turn"
            try:
                result = agentknit.run_turn(
                    client,
                    self.session["model"],
                    self.session,
                    text,
                    cancel=self.cancel_token,
                )
                self.message_count += 2
                if result.final_reply is None:
                    stop_reason = "cancelled"
            except KeyboardInterrupt:
                stop_reason = "cancelled"
            except Exception as exc:
                _log("run_turn error:\n" + traceback.format_exc())
                _notify(self.id, {
                    "sessionUpdate": "agent_message_chunk",
                    "messageId": uuid.uuid4().hex,
                    "content": {"type": "text", "text": f"\n\n[turn failed] {exc}"},
                })
                stop_reason = "error"
            finally:
                self.cancel_token = None
            return stop_reason

    def cancel(self) -> None:
        token = self.cancel_token
        if token is not None:
            token.cancel()

    def set_face(self, face_id: str) -> list[dict]:
        face = next((f for f in FACES if f["id"] == face_id), None)
        if face is None:
            raise ValueError(f"unknown model: {face_id}")
        self.face = face
        self.schema = self._schema()
        old_cwd = os.getcwd()
        try:
            if self.cwd and Path(self.cwd).is_dir():
                os.chdir(self.cwd)
            self.session = agentknit.init_session(
                self.schema,
                non_interactive=True,
                session=self.session,
            )
        finally:
            os.chdir(old_cwd)
        return config_options(self.face["id"])


SESSIONS: dict[str, FaceSession] = {}
TITLES_PATH = LOG_DIR / "titles.json"
# Index of every session this adapter ever created. It is what lets
# session/list survive a daemon restart: harness-remote's bridge re-lists
# sessions on startup and only re-exposes sessions the adapter still knows.
INDEX_PATH = LOG_DIR / "sessions.json"


def _load_index() -> list[dict]:
    try:
        return json.loads(INDEX_PATH.read_text())
    except Exception:
        return []


def _save_index_entry(session_id: str, cwd: str, agentknit_session_id: str) -> None:
    entries = _load_index()
    if not any(e["sessionId"] == session_id for e in entries):
        entries.append({"sessionId": session_id, "cwd": cwd,
                        "agentknitSessionId": agentknit_session_id})
        tmp = INDEX_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=1))
        tmp.replace(INDEX_PATH)


def _now_ms() -> int:
    import time
    return int(time.time() * 1000)


def _load_titles() -> dict:
    try:
        return json.loads(TITLES_PATH.read_text())
    except Exception:
        return {}


def _save_titles(titles: dict) -> None:
    TITLES_PATH.write_text(json.dumps(titles, indent=1))


def config_options(current: str) -> list[dict]:
    return [{
        "id": "model",
        "name": "Model",
        "category": "model",
        "type": "select",
        "currentValue": current,
        "options": [{"value": f["id"], "name": f["name"]} for f in FACES],
    }]


def _discover_agentknit_sessions() -> list[dict]:
    """List persisted agentknit sessions from the shared log base."""
    base = Path.home() / ".local/share/agent_probe"
    out = []
    titles = _load_titles()
    if not base.is_dir():
        return out
    for path in sorted(base.glob("*/*_messages.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        sid = path.name[: -len("_messages.json")]
        try:
            messages = json.loads(path.read_text())
            if not isinstance(messages, list):
                continue
            first_user = next((m for m in messages if m.get("role") == "user"), None)
            title = titles.get(sid)
            if not title and first_user:
                content = first_user.get("content", "")
                if isinstance(content, list):
                    content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
                title = re.sub(r"\s+", " ", str(content))[:80]
            out.append({
                "sessionId": f"ak:{path.parent.name}:{sid}",
                "cwd": str(Path.home()),
                "title": title or sid,
                "updatedAt": _iso(path.stat().st_mtime),
                "_meta": {"messageCount": len(messages)},
            })
        except Exception:
            continue
        if len(out) >= 100:
            break
    return out


def _iso(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()


def handle_request(req_id, method: str, params: dict) -> None:
    if method == "initialize":
        _respond(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "agentInfo": {"name": "agentknit-acp", "title": "agentknit", "version": "0.1.0"},
            "agentCapabilities": {
                "promptCapabilities": {"image": False, "embeddedContext": False},
                "sessionCapabilities": {"resume": {}, "list": {}},
                "loadSession": False,
            },
            "authMethods": [],
        })
        return

    if method == "authenticate":
        _respond(req_id, {})
        return

    if method == "session/new":
        cwd = params.get("cwd") or str(Path.home())
        sid = uuid.uuid4().hex[:12]
        fs = FaceSession(sid, cwd, FACES[0])
        SESSIONS[sid] = fs
        _save_index_entry(sid, cwd, fs.session["session_id"])
        _respond(req_id, {"sessionId": sid, "configOptions": config_options(fs.face["id"])})
        return

    if method == "session/list":
        titles = _load_titles()
        sessions = []
        for entry in _load_index():
            sid = entry["sessionId"]
            fs = SESSIONS.get(sid)
            sessions.append({
                "sessionId": sid,
                "cwd": entry.get("cwd") or str(Path.home()),
                "title": (fs.title if fs and fs.title else titles.get(sid)) or sid,
                "updatedAt": _iso((fs.updated / 1000) if fs else INDEX_PATH.stat().st_mtime),
                "_meta": {"messageCount": fs.message_count if fs else 0},
            })
        sessions.extend(_discover_agentknit_sessions())
        _respond(req_id, {"sessions": sessions})
        return

    if method in ("session/resume", "session/load"):
        sid = params.get("sessionId", "")
        fs = SESSIONS.get(sid)
        if fs is None:
            entry = next((e for e in _load_index() if e["sessionId"] == sid), None)
            if entry is None:
                _respond(req_id, error={"code": -32602, "message": f"unknown session: {sid}"})
                return
            # Lazily re-materialize after an adapter restart. The agentknit
            # conversation resumes from its persisted journal/snapshot via
            # resumed_from. The ACP session id is 12 hex chars; the agentknit
            # session was created with its own random id, so map through the
            # durable index written at session/new time.
            fs = FaceSession.__new__(FaceSession)
            fs.id = sid
            agentknit_sid = entry.get("agentknitSessionId", sid)
            fs.cwd = params.get("cwd") or entry.get("cwd") or str(Path.home())
            fs.face = FACES[0]
            fs.lock = threading.Lock()
            fs.cancel_token = None
            fs.turn_thread = None
            fs.title = _load_titles().get(sid)
            fs.created = _now_ms()
            fs.updated = fs.created
            fs.message_count = 0
            fs._first_chunk_ids = {}
            fs.schema = fs._schema()
            old_cwd = os.getcwd()
            try:
                if fs.cwd and Path(fs.cwd).is_dir():
                    os.chdir(fs.cwd)
                fs.session = agentknit.init_session(
                    fs.schema,
                    non_interactive=True,
                    resumed_from=agentknit_sid,
                    min_cacheable_tokens=int(os.environ.get("AGENTKNIT_ACP_MIN_CACHEABLE", "1024")),
                )
                fs.message_count = max(0, len(fs.session.get("messages", [])) - 1)
            finally:
                os.chdir(old_cwd)
            SESSIONS[sid] = fs
        _respond(req_id, {"configOptions": config_options(fs.face["id"])})
        return

    if method == "session/prompt":
        sid = params.get("sessionId", "")
        fs = SESSIONS.get(sid)
        if fs is None:
            _respond(req_id, error={"code": -32602, "message": f"unknown session: {sid}"})
            return
        parts = params.get("prompt") or []
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
        if not text:
            _respond(req_id, error={"code": -32602, "message": "empty prompt"})
            return
        if fs.title is None:
            fs.title = re.sub(r"\s+", " ", text)[:80]
            titles = _load_titles()
            titles[fs.id] = fs.title
            _save_titles(titles)

        def work() -> None:
            stop_reason = fs.prompt(text)
            _respond(req_id, {"stopReason": stop_reason})

        fs.turn_thread = threading.Thread(target=work, daemon=True)
        fs.turn_thread.start()
        return  # response sent when the turn finishes

    if method == "session/set_config_option":
        sid = params.get("sessionId", "")
        fs = SESSIONS.get(sid)
        if fs is None:
            _respond(req_id, error={"code": -32602, "message": f"unknown session: {sid}"})
            return
        config_id = params.get("configId")
        value = params.get("value")
        if config_id == "model":
            try:
                _respond(req_id, {"configOptions": fs.set_face(value)})
            except ValueError as exc:
                _respond(req_id, error={"code": -32602, "message": str(exc)})
            return
        _respond(req_id, error={"code": -32602, "message": f"unknown config option: {config_id}"})
        return

    _respond(req_id, error={"code": -32601, "message": f"method not found: {method}"})


def handle_notification(method: str, params: dict) -> None:
    if method == "session/cancel":
        sid = params.get("sessionId", "")
        fs = SESSIONS.get(sid)
        if fs is not None:
            fs.cancel()


def main() -> None:
    _log(f"agentknit-acp starting: model={MODEL} endpoint={ENDPOINT}")
    buffer = ""
    while True:
        chunk = sys.stdin.read(1)
        if chunk == "":
            break
        if chunk != "\n":
            buffer += chunk
            continue
        line = buffer.strip()
        buffer = ""
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method", "")
        params = msg.get("params") or {}
        req_id = msg.get("id")
        try:
            if req_id is not None:
                handle_request(req_id, method, params)
            else:
                handle_notification(method, params)
        except Exception as exc:
            _log("dispatch error:\n" + traceback.format_exc())
            if req_id is not None:
                _respond(req_id, error={"code": -32603, "message": str(exc)})
    _log("agentknit-acp stdin closed, exiting")


if __name__ == "__main__":
    main()
