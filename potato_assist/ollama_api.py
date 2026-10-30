"""Ollama-compatible API, so Home Assistant's built-in Ollama integration can use the Potato Assist engines.

The integration (ollama-python 0.6.2) calls only: GET /api/tags, POST /api/pull (blocking), POST /api/chat
(streamed NDJSON, with tools, images, think, format and options). /api/show, /api/version and /api/ps are
answered for other Ollama clients. Chat is translated to llama-server's OpenAI-style /v1/chat/completions.
"""

import asyncio
import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from aiohttp import web

import grouped_context
from json_schema_to_grammar import SchemaConverter

OLLAMA_VERSION = "0.12.6"
NDJSON = "application/x-ndjson"
HA_API = "http://supervisor/core/api"  # reachable with the app's SUPERVISOR_TOKEN (homeassistant_api: true)
SEEDABLE_ENGINES = ("ik", "millie", "llama")  # all serve hybrid models that resume prompts from checkpoints
# Mainline llama.cpp checkpoints the start of the last user message itself, so a request needs no seeding; a
# separate seed would re-read from an earlier checkpoint and invalidate that one. It is only warmed on load.
NATIVE_CHECKPOINT_ENGINES = ("llama",)


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ollama_name(model_id):
    return f"{model_id}:latest"


def model_id_from(name, models):
    """Accept 'bonsai-2-9b', 'bonsai-2-9b:latest' and any other tag."""
    base = (name or "").split(":", 1)[0]
    return base if base in models else None


def image_url(b64):
    head = b64[:8]
    mime = ("image/png" if head.startswith("iVBOR") else "image/gif" if head.startswith("R0lG")
            else "image/webp" if head.startswith("UklGR") else "image/jpeg")
    return f"data:{mime};base64,{b64}"


def to_openai_messages(messages, grouped=False):
    """Ollama chat messages -> OpenAI messages. Ollama tool results carry no call id, so ids are assigned to
    the assistant's tool calls and handed to the following tool messages in order. With grouped, the Static and
    Live Context are grouped by area (grouped_context) instead of the one-line-per-entity compact form."""
    out, pending_ids, counter = [], [], 0
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user" and m.get("images"):
            parts = [{"type": "text", "text": content}] if content else []
            parts += [{"type": "image_url", "image_url": {"url": image_url(img)}} for img in m["images"]]
            out.append({"role": "user", "content": parts})
        elif role == "assistant" and m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                counter += 1
                call_id = f"call_{counter}"
                pending_ids.append(call_id)
                fn = call.get("function", {})
                args = fn.get("arguments", {})
                calls.append({"id": call_id, "type": "function", "function": {
                    "name": fn.get("name", ""), "arguments": args if isinstance(args, str) else json.dumps(args)}})
            out.append({"role": "assistant", "content": content, "tool_calls": calls})
        elif role == "tool":
            msg = {"role": "tool", "content": (grouped_context.rewrite_tool_content(content) if grouped
                                               else compact_tool_result(content))}
            if pending_ids:
                msg["tool_call_id"] = pending_ids.pop(0)
            out.append(msg)
        elif role == "system" and grouped:
            out.append({"role": role, "content": grouped_context.rewrite_system(content)})
        else:
            out.append({"role": role, "content": content})
    return out


def yaml_scalar(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1].replace("''", "'")
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1]
    return text


def compact_live_context(text):
    """HA's GetLiveContext YAML -> one line per entity, about a third of the tokens. Returns None for anything
    that is not the plain shape HA emits (names/domain/state/areas plus one level of attributes)."""
    head, _, body = text.partition("\n")
    entities, cur, in_attrs = [], None, False
    for line in body.splitlines():
        if not line.strip():
            continue
        if line.startswith("- names: "):
            cur = {"name": yaml_scalar(line[9:]), "attrs": []}
            entities.append(cur)
            in_attrs = False
        elif cur is None:
            return None
        elif line == "  attributes:":
            in_attrs = True
        elif in_attrs and line.startswith("    ") and not line.startswith("     "):
            key, sep, value = line.strip().partition(":")
            if not sep:
                return None
            if value.strip() and key != "device_class":
                cur["attrs"].append((key, yaml_scalar(value)))
        elif line.startswith("  ") and not line.startswith("   "):
            key, sep, value = line.strip().partition(":")
            if key not in ("domain", "state", "areas") or not sep:
                return None
            cur[key] = yaml_scalar(value)
            in_attrs = False
        else:
            return None
    if not entities:
        return None
    lines = [head]
    for e in entities:
        where = ", ".join(x for x in (e.get("domain"), e.get("areas")) if x)
        attrs = dict(e["attrs"])
        state, unit = e.get("state", ""), attrs.pop("unit_of_measurement", "")
        value = state if state in ("unavailable", "unknown") else " ".join(x for x in (state, unit) if x)
        extra = "; " + ", ".join(f"{k} {v}" for k, v in attrs.items()) if attrs else ""
        lines.append(f"- {e['name']} ({where}): {value}{extra}")
    return "\n".join(lines)


def compact_tool_result(content):
    """HA sends tool results as JSON ({"data": {"result": ...}, "error": ...}); the live-context one is YAML
    inside it, with every newline and quote escaped. Small models read the compact form faster and as well."""
    try:
        obj = json.loads(content)
    except ValueError:
        return content
    if not isinstance(obj, dict) or obj.get("error"):
        return content
    data = obj.get("data", obj)
    result = data.get("result") if isinstance(data, dict) else None
    if isinstance(result, str) and result.startswith("Live Context:"):
        return compact_live_context(result) or content
    return content


def action_targets(content):
    """Entity ids an intent tool result reports as done ({"data": {"response_type": "action_done",
    "data": {"success": [{"type": "entity", "id": ...}]}}}); [] for any other result."""
    try:
        obj = json.loads(content)
    except ValueError:
        return []
    data = obj.get("data") if isinstance(obj, dict) else None
    if not isinstance(data, dict) or data.get("response_type") != "action_done" or not isinstance(data.get("data"), dict):
        return []
    return [t["id"] for t in data["data"].get("success") or []
            if isinstance(t, dict) and t.get("type") == "entity" and isinstance(t.get("id"), str)]


def mark_unavailable(content, unavailable):
    """HA reports a command to an offline (unavailable) entity as done, and the model then tells the user it
    worked. Move those entities to the failed list; if nothing succeeded, return HA's error shape instead."""
    obj = json.loads(content)
    inner = obj["data"]["data"]
    bad = [t for t in inner["success"] if t.get("id") in unavailable]
    good = [t for t in inner["success"] if t.get("id") not in unavailable]
    # An area or floor command lists the area/floor itself first; it says nothing about whether an entity changed.
    if not any(t.get("type") == "entity" for t in good) and not inner.get("failed"):
        names = [t.get("name") or t["id"] for t in bad]
        names = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
        verb = "is" if len(bad) == 1 else "are"
        return json.dumps({"data": {"error": "EntityUnavailable",
                                    "error_text": f"{names} {verb} unavailable (offline), so nothing changed"},
                           "error": True}, ensure_ascii=False, separators=(",", ":"))
    inner = {**inner, "success": good, "failed": (inner.get("failed") or []) + [{**t, "error": "unavailable"} for t in bad]}
    return json.dumps({**obj, "data": {**obj["data"], "data": inner}}, ensure_ascii=False, separators=(",", ":"))


def schema_grammar(schema):
    """JSON schema -> GBNF. Sent as a raw grammar because ik's response_format path wraps the schema in a
    chat-template rule ('<|im_start|>assistant...') that fails grammar init for the Qwen3.5 template."""
    conv = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
    conv.visit(conv.resolve_refs(schema, "input"), "")
    return conv.format_grammar()


def unfiltered_live_context(tool):
    """Offer GetLiveContext without its filter parameters. The guard drops filters anyway (see
    guard_tool_call), and writing them cost the 2B ~40 extra tokens (~5 s) per state question."""
    fn = tool.get("function") or {}
    if not fn.get("name", "").endswith("GetLiveContext"):
        return tool
    desc = fn.get("description", "").split(" You may filter")[0]
    return {**tool, "function": {**fn, "description": desc,
                                 "parameters": {"type": "object", "properties": {}}}}


FORCE_ANSWER = "Answer now from the tool results above. Do not call another tool."
MAX_TOOL_ROUNDS = 3


def tool_loop(messages):
    """True when this turn (since the last user message) repeats a tool call with the same arguments, or has
    reached MAX_TOOL_ROUNDS tool rounds: the model is going round in circles, and HA would keep executing
    the calls until its own limit (LFM3 took 30-60 s on such requests)."""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    seen, rounds = set(), 0
    for m in messages[last_user + 1:]:
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        rounds += 1
        for call in m["tool_calls"]:
            fn = call.get("function", {})
            try:
                args = json.dumps(json.loads(fn.get("arguments") or "{}"), sort_keys=True)
            except ValueError:
                args = fn.get("arguments")
            key = (fn.get("name"), args)
            if key in seen:
                return True
            seen.add(key)
    return rounds >= MAX_TOOL_ROUNDS


def to_openai_request(body, grouped=False):
    opts = body.get("options") or {}
    req = {"messages": to_openai_messages(body.get("messages") or [], grouped), "stream": True,
           "chat_template_kwargs": {"enable_thinking": bool(body.get("think"))}}
    if body.get("tools"):
        req["tools"] = [unfiltered_live_context(t) for t in body["tools"]]
    for ollama_key, openai_key in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                                   ("seed", "seed"), ("stop", "stop"), ("num_predict", "max_tokens"),
                                   ("repeat_penalty", "repeat_penalty"), ("presence_penalty", "presence_penalty")):
        if opts.get(ollama_key) is not None:
            req[openai_key] = opts[ollama_key]
    fmt = body.get("format")
    if fmt == "json" or isinstance(fmt, dict):
        req["grammar"] = schema_grammar(fmt if isinstance(fmt, dict) else {"type": "object"})
        req["chat_template_kwargs"]["enable_thinking"] = False  # the grammar would forbid thinking text
    return req


class ThinkSplitter:
    """Splits streamed content into thinking and answer text. Bonsai (ik, --reasoning-format none) streams
    '<think>...</think>' inline, often empty when thinking is off."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.mode = "start"  # start | think | answer
        self.buf = ""

    def feed(self, text):
        """Returns (thinking, answer) text that is safe to emit."""
        self.buf += text
        thinking = answer = ""
        while True:
            if self.mode == "start":
                stripped = self.buf.lstrip()
                if stripped.startswith(self.OPEN):
                    self.buf = stripped[len(self.OPEN):]
                    self.mode = "think"
                elif not stripped or self.OPEN.startswith(stripped):
                    return thinking, answer  # could still be the opening tag
                else:
                    self.mode = "answer"
            if self.mode == "think":
                end = self.buf.find(self.CLOSE)
                if end < 0:
                    keep = len(self.CLOSE) - 1  # a closing tag may be split across chunks
                    thinking += self.buf[:-keep] if len(self.buf) > keep else ""
                    self.buf = self.buf[-keep:] if len(self.buf) > keep else self.buf
                    return thinking, answer
                thinking += self.buf[:end]
                self.buf = self.buf[end + len(self.CLOSE):].lstrip()
                self.mode = "answer"
            if self.mode == "answer":
                answer += self.buf
                self.buf = ""
                return thinking, answer

    def flush(self):
        rest, self.buf = self.buf, ""
        if self.mode == "think":
            return rest, ""
        return "", rest.lstrip() if self.mode == "start" else rest


def guard_tool_call(name, args):
    """Correct tool calls small models get wrong, before Home Assistant runs them.

    GetLiveContext: small models put device names into `domain`/`area`/`name` filters and then get nothing
    back. Unfiltered, HA returns every exposed entity with its state; with a few dozen entities that is cheap
    and always contains the answer, so filters are dropped.
    """
    if name.endswith("GetLiveContext") and args:
        return {}, f"dropped filters {json.dumps(args)}"
    return args, None


def chunk(model, message, done=False, **extra):
    return {"model": model, "created_at": now_iso(), "message": {"role": "assistant", **message}, "done": done, **extra}


def ollama_error(message, status=500):
    return web.json_response({"error": message}, status=status)


class OllamaAPI:
    def __init__(self, app, models, engine, model_file, mmproj_file, clean_params, settings_params):
        self.app = app
        self.models = models
        self.engine = engine
        self.model_file = model_file
        self.mmproj_file = mmproj_file
        self.clean_params = clean_params
        self.settings_params = settings_params  # callable: model_id -> saved params
        self._switch = asyncio.Lock()
        self._pinned = None  # (model, load time, prefix) the millie engine holds a pinned checkpoint for
        self._seeding = asyncio.Lock()  # a warm-up and HA's first request must not seed the same prefix twice
        engine.on_ready.append(self.warm_prefix)

    def register(self, router):
        router.add_get("/api/version", self.version)
        router.add_get("/api/tags", self.tags)
        router.add_get("/api/ps", self.ps)
        router.add_post("/api/show", self.show)
        router.add_post("/api/pull", self.pull)
        router.add_post("/api/chat", self.chat)

    # ------------------------------------------------------------ metadata

    def files_for(self, model_id):
        params = self.settings_params(model_id)
        files = [self.model_file(model_id, params)]
        if params.get("vision"):
            files.append(self.mmproj_file(model_id))
        return params, files

    def downloaded(self, model_id):
        _, files = self.files_for(model_id)
        return all((self.engine.model_dir / f[1]).exists() for f in files)

    def details(self, model_id):
        m = self.models[model_id]
        params, _ = self.files_for(model_id)
        size = m.get("size") or "35B-A3B"
        family = m.get("family") or ("qwen35moe" if m["engine"] == "millie" else "qwen35")
        quant = m.get("quant") or params.get("format", "QT_MS32K4")
        return {"parent_model": "", "format": "gguf", "family": family, "families": [family],
                "parameter_size": size, "quantization_level": quant}

    def entry(self, model_id):
        _, files = self.files_for(model_id)
        main = self.engine.model_dir / files[0][1]
        modified = datetime.fromtimestamp(main.stat().st_mtime, timezone.utc).isoformat() if main.exists() else now_iso()
        return {"name": ollama_name(model_id), "model": ollama_name(model_id), "modified_at": modified,
                "size": sum(f[3] for f in files), "digest": files[0][2], "details": self.details(model_id)}

    async def version(self, request):
        return web.json_response({"version": OLLAMA_VERSION})

    async def tags(self, request):
        return web.json_response({"models": [self.entry(mid) for mid in self.models if self.downloaded(mid)]})

    async def ps(self, request):
        e = self.engine
        if e.state != "ready" or not e.model:
            return web.json_response({"models": []})
        item = self.entry(e.model)
        item.update({"expires_at": "2318-01-01T00:00:00Z", "size_vram": 0})
        return web.json_response({"models": [item]})

    async def show(self, request):
        body = await request.json()
        model_id = model_id_from(body.get("model") or body.get("name"), self.models)
        if not model_id:
            return ollama_error("model not found", 404)
        params, _ = self.files_for(model_id)
        caps = ["completion", "tools"] + (["vision"] if params.get("vision") else []) + ["thinking"]
        return web.json_response({"modelfile": "", "parameters": "", "template": "", "license": "",
                                  "details": self.details(model_id), "model_info": {}, "capabilities": caps,
                                  "modified_at": self.entry(model_id)["modified_at"]})

    # ------------------------------------------------------------ pull

    async def pull(self, request):
        body = await request.json()
        model_id = model_id_from(body.get("model") or body.get("name"), self.models)
        if not model_id:
            return ollama_error(f"pull model manifest: file does not exist ({body.get('model')})", 404)
        _, files = self.files_for(model_id)
        task = asyncio.create_task(self.engine.fetch_files(files))
        if body.get("stream") is False:
            try:
                await task
            except Exception as err:  # surfaced to the client as an Ollama error
                return ollama_error(str(err))
            return web.json_response({"status": "success"})
        resp = web.StreamResponse(headers={"Content-Type": NDJSON})
        await resp.prepare(request)
        await resp.write(json.dumps({"status": "pulling manifest"}).encode() + b"\n")
        while not task.done():
            d = self.engine.download
            if d:
                line = {"status": f"pulling {d['file']}", "digest": f"sha256:{d['file']}",
                        "total": d["total"], "completed": d["done"]}
                await resp.write(json.dumps(line).encode() + b"\n")
            await asyncio.sleep(1)
        try:
            task.result()
            await resp.write(json.dumps({"status": "success"}).encode() + b"\n")
        except Exception as err:
            await resp.write(json.dumps({"error": str(err)}).encode() + b"\n")
        await resp.write_eof()
        return resp

    # ------------------------------------------------------------ chat

    async def ensure_model(self, model_id):
        """Switch the single engine to the requested model if needed and wait until it serves."""
        async with self._switch:
            e = self.engine
            if e.model != model_id or e.state in ("stopped", "error"):
                await e.load(model_id, self.settings_params(model_id))
            while e.state in ("downloading", "loading"):
                await asyncio.sleep(0.5)
            if e.state != "ready":
                raise RuntimeError(e.message or f"model {model_id} failed to load ({e.state})")

    # ------------------------------------------------------------ prefix seeding
    #
    # Bonsai and Millie are hybrid (recurrent) models: the engine can resume a cached prompt only from a saved checkpoint,
    # and it saves one 5 tokens before the end of each prompt and at batch boundaries (every 2048 tokens).
    # HA's system prompt and tools are ~2,650 identical tokens on every request, but a new conversation
    # diverges right after them, so without help it resumes from token 2048 and re-reads ~600 tokens (~8 s).
    # Sending the shared prefix alone first leaves a checkpoint at its end; the real request then reads only
    # the user's message. The prefix is also kept on disk and replayed when a model loads.

    def grouped(self, model_id):
        return self.models[model_id].get("context") == "grouped"

    def prefix_file(self, model_id):
        """The last HA prefix as this model's context format renders it."""
        name = "ha-prefix-grouped.json" if self.grouped(model_id) else "ha-prefix.json"
        return self.engine.model_dir.parent / name

    def seedable(self, req):
        roles = [m["role"] for m in req["messages"]]
        return (self.models[self.engine.model]["engine"] in SEEDABLE_ENGINES and req.get("tools")
                and roles[-1:] == ["user"] and not {"assistant", "tool"} & set(roles))

    async def seed_prefix(self, template_req):
        """Process the rendered prompt up to the last user turn, so its end is checkpointed."""
        async with self._seeding:
            return await self._seed_prefix(template_req)

    async def _seed_prefix(self, template_req):
        session, url = self.app["client"], self.engine.url
        if self.models[self.engine.model]["engine"] in NATIVE_CHECKPOINT_ENGINES:
            # The prefix as an ordinary chat turn: the engine leaves a checkpoint where the user message starts.
            started = time.monotonic()
            body = {**template_req, "max_tokens": 1, "temperature": 0}
            async with session.post(f"{url}/v1/chat/completions", json=body) as r:
                timings = (await r.json()).get("timings", {}) if r.status == 200 else {}
            return timings.get("prompt_n"), round(time.monotonic() - started, 2)
        async with session.post(f"{url}/apply-template", json=template_req) as r:
            if r.status != 200:
                return None
            prompt = (await r.json())["prompt"]
        cut = prompt.rfind("<|im_start|>user")
        if cut <= 0:
            return None
        body = {"prompt": prompt[:cut], "n_predict": 1, "cache_prompt": True, "temperature": 0}
        key = None
        if self.models[self.engine.model]["engine"] == "millie":
            # millie-runtime's server cannot resume a fully cached prompt (it steps back one token, fails to
            # truncate the recurrent state and re-reads everything, ~230 s), so seed each prefix once per load,
            # into a pinned checkpoint that later requests' checkpoints never evict.
            key = (self.engine.model, self.engine.loaded_at, body["prompt"])
            if key == self._pinned:
                return "already seeded"
            body["pin_checkpoint_from_end"] = 4
        started = time.monotonic()
        async with session.post(f"{url}/completion", json=body) as r:
            timings = (await r.json()).get("timings", {}) if r.status == 200 else {}
        if key and timings:
            self._pinned = key
        return timings.get("prompt_n"), round(time.monotonic() - started, 2)

    def remember_prefix(self, req):
        keep = {"messages": [m for m in req["messages"] if m["role"] == "system"] + [{"role": "user", "content": "."}],
                "tools": req["tools"], "chat_template_kwargs": req.get("chat_template_kwargs", {})}
        text = json.dumps(keep)
        path = self.prefix_file(self.engine.model)
        try:
            if not path.exists() or path.read_text() != text:
                path.write_text(text)
        except OSError as err:
            self.engine.logs.append(f"[ollama] could not save the HA prefix: {err}")
        return keep

    async def warm_prefix(self):
        """Called when an engine becomes ready: pre-process the last HA prefix before anyone asks."""
        try:
            path = self.prefix_file(self.engine.model)
            if self.models[self.engine.model]["engine"] not in SEEDABLE_ENGINES or not path.exists():
                return
            with self.engine.busy_with("Preparing the Home Assistant prompt"):
                result = await self.seed_prefix(json.loads(path.read_text()))
            self.engine.logs.append(f"[ollama] warmed the HA prefix (tokens read, seconds): {result}")
        except Exception as err:  # warming is an optimisation; never break the engine over it
            self.engine.logs.append(f"[ollama] prefix warm-up failed: {err}")

    async def check_unavailable(self, session, messages):
        """Rewrite this turn's action results whose targets are unavailable in HA (see mark_unavailable).
        Needs the app's HA API access; without it, or if HA does not answer, results pass through unchanged."""
        token = os.environ.get("SUPERVISOR_TOKEN")
        last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
        for m in messages[last_user + 1:]:
            if m.get("role") != "tool" or not token:
                continue
            unavailable = set()
            for entity_id in action_targets(m.get("content") or ""):
                try:
                    async with session.get(f"{HA_API}/states/{entity_id}", timeout=aiohttp.ClientTimeout(total=3),
                                           headers={"Authorization": f"Bearer {token}"}) as r:
                        if r.status == 200 and (await r.json()).get("state") == "unavailable":
                            unavailable.add(entity_id)
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
                    self.engine.logs.append(f"[ollama] state check for {entity_id} failed: {err}")
            if unavailable:
                m["content"] = mark_unavailable(m["content"], unavailable)
                self.engine.logs.append(f"[ollama] reported unavailable targets as failed: {sorted(unavailable)}")

    async def chat(self, request):
        with self.engine.busy_with("Answering a Home Assistant request"):
            return await self._chat(request)

    async def _chat(self, request):
        body = await request.json()
        # Debug aid: a debug-ollama file in the share keeps the last request (system prompt, tools) for inspection.
        share = self.engine.model_dir.parent
        if (share / "debug-ollama").exists():
            (share / "last-ollama-request.json").write_text(json.dumps(body, indent=1))
        name = body.get("model")
        model_id = model_id_from(name, self.models)
        if not model_id:
            return ollama_error(f"model '{name}' not found", 404)
        try:
            await self.ensure_model(model_id)
        except Exception as err:
            return ollama_error(str(err))
        want_thinking = bool(body.get("think"))
        stream = body.get("stream", True)
        session = self.app["client"]
        await self.check_unavailable(session, body.get("messages") or [])
        req = to_openai_request(body, self.grouped(model_id))
        if self.models[model_id]["engine"] in NATIVE_CHECKPOINT_ENGINES and req.get("tools") \
                and tool_loop(req["messages"]):
            # Appended after everything already cached, so forcing the answer costs only the new note's tokens.
            req["tool_choice"] = "none"
            req["messages"].append({"role": "user", "content": FORCE_ANSWER})
            self.engine.logs.append("[ollama] tool loop: asked for an answer without tools")
        started = time.monotonic_ns()
        if self.seedable(req):
            try:
                prefix = self.remember_prefix(req)
                if self.models[model_id]["engine"] not in NATIVE_CHECKPOINT_ENGINES:
                    result = await self.seed_prefix(prefix)
                    self.engine.logs.append(f"[ollama] seeded the HA prefix (tokens read, seconds): {result}")
            except Exception as err:
                self.engine.logs.append(f"[ollama] prefix seeding failed: {err}")

        resp = None
        if stream:
            resp = web.StreamResponse(headers={"Content-Type": NDJSON})
            await resp.prepare(request)

        async def emit(obj):
            if resp is not None:
                await resp.write(json.dumps(obj).encode() + b"\n")

        splitter = ThinkSplitter()
        content_all, thinking_all = [], []
        calls = {}  # index -> {"name", "arguments"}
        timings, finish = {}, None
        try:
            async with session.post(f"{self.engine.url}/v1/chat/completions", json=req) as upstream:
                if upstream.status != 200:
                    text = await upstream.text()
                    if resp is not None:
                        await emit({"error": text[:500]})
                        await resp.write_eof()
                        return resp
                    return ollama_error(text[:500], upstream.status)
                buffer = b""
                async for raw in upstream.content.iter_any():
                    buffer += raw
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            continue
                        event = json.loads(data)
                        if event.get("timings"):
                            timings = event["timings"]
                        for choice in event.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if choice.get("finish_reason"):
                                finish = choice["finish_reason"]
                            for tc in delta.get("tool_calls") or []:
                                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                                fn = tc.get("function") or {}
                                slot["name"] += fn.get("name") or ""
                                slot["arguments"] += fn.get("arguments") or ""
                            think_text, answer_text = "", ""
                            if delta.get("reasoning_content"):
                                think_text += delta["reasoning_content"]
                            if delta.get("content"):
                                t, a = splitter.feed(delta["content"])
                                think_text += t
                                answer_text += a
                            if think_text:
                                thinking_all.append(think_text)
                                if want_thinking:
                                    await emit(chunk(name, {"content": "", "thinking": think_text}))
                            if answer_text:
                                content_all.append(answer_text)
                                await emit(chunk(name, {"content": answer_text}))
        except (ConnectionResetError, asyncio.CancelledError):
            raise
        t, a = splitter.flush()
        if t:
            thinking_all.append(t)
            if want_thinking:
                await emit(chunk(name, {"content": "", "thinking": t}))
        if a:
            content_all.append(a)
            await emit(chunk(name, {"content": a}))

        tool_calls = []
        for _, call in sorted(calls.items()):
            try:
                args = json.loads(call["arguments"]) if call["arguments"] else {}
            except ValueError:
                args = {"raw": call["arguments"]}
            args, note = guard_tool_call(call["name"], args)
            if note:
                self.engine.logs.append(f"[ollama] guard {call['name']}: {note}")
            tool_calls.append({"function": {"name": call["name"], "arguments": args}})
        if tool_calls:
            self.engine.logs.append(f"[ollama] tool calls: {json.dumps(tool_calls)}")
            await emit(chunk(name, {"content": "", "tool_calls": tool_calls}))

        total = time.monotonic_ns() - started
        stats = {"done_reason": "stop" if finish in (None, "stop", "tool_calls") else finish,
                 "total_duration": total, "load_duration": 0,
                 "prompt_eval_count": timings.get("prompt_n", 0),
                 "prompt_eval_duration": int(timings.get("prompt_ms", 0) * 1e6),
                 "eval_count": timings.get("predicted_n", 0),
                 "eval_duration": int(timings.get("predicted_ms", 0) * 1e6)}
        if resp is not None:
            await emit(chunk(name, {"content": ""}, done=True, **stats))
            await resp.write_eof()
            return resp
        message = {"content": "".join(content_all)}
        if want_thinking and thinking_all:
            message["thinking"] = "".join(thinking_all)
        if tool_calls:
            message["tool_calls"] = tool_calls
        return web.json_response(chunk(name, message, done=True, **stats))
