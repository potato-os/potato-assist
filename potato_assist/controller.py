"""Potato Assist controller: one model at a time, three engines, Pi telemetry.

Serves the chat UI and a small API on :8080 (ingress and LAN), runs llama-server on 127.0.0.1:8090,
and proxies the OpenAI-compatible routes to it. Engines:
  ik      ik_llama.cpp 804ee29        Bonsai 2 (PQ2_0 / Q2_0)
  millie  millie-runtime 8b55b87 + Golem patch series 520266146a39 (hot experts on the CPU)
  llama   llama.cpp eb4e244           LFM3 / LFM3-VL (not yet in ik_llama.cpp)
"""

import asyncio
import contextlib
import copy
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import socket
import struct
import time
from collections import deque
from pathlib import Path

import aiohttp
from aiohttp import web

from ollama_api import OllamaAPI

log = logging.getLogger("potato-assist")

# Paths are overridable so the controller can be exercised off the Pi.
DATA = Path(os.environ.get("POTATO_DATA", "/data"))
SETTINGS_FILE = DATA / "settings.json"
OPTIONS_FILE = DATA / "options.json"
SHARE = Path(os.environ.get("POTATO_SHARE", "/share/potato-assist"))  # models and debug files, visible over Samba
MODEL_DIR = SHARE / "models"
WWW = Path(os.environ.get("POTATO_WWW", "/www"))
ENGINE_ROOT = os.environ.get("POTATO_ENGINES", "/opt/engines")
ENGINE_HOST = "127.0.0.1"
LISTEN_PORT = int(os.environ.get("POTATO_PORT", "8080"))


def free_port():
    """A fresh loopback port per engine start: the previous engine's closed connections can hold its port in
    TIME_WAIT for a while, and llama-server does not reuse a busy port."""
    with socket.socket() as s:
        s.bind((ENGINE_HOST, 0))
        return s.getsockname()[1]

ENGINES = {
    "ik": {"label": "ik_llama.cpp 804ee29", "server": f"{ENGINE_ROOT}/ik/llama-server"},
    "millie": {"label": "millie-runtime 8b55b87 + Golem 520266146a39", "server": f"{ENGINE_ROOT}/millie/llama-server"},
    "llama": {"label": "llama.cpp eb4e244", "server": f"{ENGINE_ROOT}/llama/llama-server"},
}

BONSAI_FILES = {
    "2B": {"PQ2_0": ("d5f01a4ecaf49239653308ee732ad1a0c5f802c0ddb997549fc7b694c8b45928", 517312992),
           "Q2_0": ("c2893fa6f951daee8b7263d5f49b455e57989ef56417a22125b03566dc560650", 546689504)},
    "4B": {"PQ2_0": ("d406b1e912c0d2a4c124ad10bf38a76ae662fdc4d24b176451403dc7b7f30409", 1146465120),
           "Q2_0": ("64a5183b44921bdb5ac4a6f5f92007821d6ebe5c66997d3c147fd214ab9433d4", 1212103520)},
    "9B": {"PQ2_0": ("7a89ab48f5038c704480f60781f9b8d4035351f5f6beb7d11fad80001fde23f9", 2416851840),
           "Q2_0": ("1f884aa56aacd2bbbb2e4b7abaead2b1102b74a9c75a23fc33dab916813a384d", 2556640128)},
}

# Vision projectors (clip.projector_type qwen3vl_merger, supported by both engines).
BONSAI_MMPROJ = {
    "2B": ("ac9934e8d7966d0d62442c202ec8fdad99a045dea79366c1a4a246a46f5c0533", 364664352),
    "4B": ("db939d5f9b7014137810205e91d05e5f54afd72b0b3af9d33bd7d82f7906a55d", 366894624),
    "9B": ("e0cb6f6988cea323ef798cd77d019bbb819af75582f034eaab0e7232314953db", 624229664),
}
MILLIE_MMPROJ = ("Millie-1.1-35B-A3B-vision.gguf",
                 "b04c6945f160a1bc4960220e7f1a6aa4cf8cbacf0cf7ec8c8881ba640ef45ba8", 295428032)

# Defaults are the winners of Pi 5 benchmark sweeps (2026-10-24/25).
# Vision is on from the start; 256 image tokens is Golem's cap (encode cost grows quadratically past it).
COMMON_PARAMS = {"threads": 4, "ctx_size": 8192, "ubatch": 128, "kv_type": "q8_0", "vision": True, "image_tokens": 256}
# Self-speculative presets (ik only): drafts come from text already in the context, no draft model. Short drafts
# suit HA's short answers and tool calls. ik disables these when vision is loaded.
SPECULATIVE = {
    "off": None,
    "ngram-mod": "ngram-mod:n_max=16,n_min=2,ngram_size_n=4",
    "ngram-simple": "ngram-simple:n_max=16",
    "ngram-map-k": "ngram-map-k:n_max=16,ngram_min_hits=1",
    "suffix": "suffix:n_max=16,n_min=2,suffix_min_match_len=3,suffix_max_depth=64",
}
SPEC_OFF = {"speculative": "off"}
MODELS = {
    # Fine-tuned for Home Assistant from LFM3-VL-2B's QAD Q4_0 file: same layout and chat template, so it shares the
    # stock vision projector. Trained and scored on the Live Context grouped by area (grouped_context).
    "lfm3-1.7b-home-assistant": {
        "label": "LFM3 1.7B Home Assistant", "engine": "llama", "size": "2B", "family": "lfm3", "quant": "Q4_0",
        "repo": "Slomin/lfm3-1.7b-home-assistant", "file": "lfm3-1.7b-home-assistant-v0.1.gguf",
        "sha256": "efd502c108a2cbb853e78da429cc098c9d2804a6c8e9af1f33e65576287f259e", "bytes": 1049615712,
        "mmproj_repo": "LiquidAI/LFM3-VL-2B-GGUF",
        "mmproj": ("mmproj-LFM3-VL-2B-Q8_0.gguf",
                   "c5c5b58d544a18da919172d2d0368090ea1a4a351b6feb7f8a326a264a5918ab", 203031872),
        "sampler": ["--temp", "0.1", "--top-k", "50", "--top-p", "1.0", "--min-p", "0", "--repeat-penalty", "1.1"],
        "context": "grouped",
        "note": "Recommended. LFM3-VL 2B fine-tuned for Home Assistant tool calls. ~6 s per Assist request on a Pi 5.",
        "flash_attn": "off",
        "defaults": {**COMMON_PARAMS, "ubatch": 128, "kv_type": "f16"},
    },
    "lfm3-vl-2b": {
        "label": "LFM3-VL 2B", "engine": "llama", "size": "2B", "family": "lfm3", "quant": "Q4_0",
        "repo": "LiquidAI/LFM3-VL-2B-GGUF", "file": "LFM3-VL-2B-QAD-Q4_0.gguf",
        "sha256": "621c455376cb96a20891697deb9975c6fb47dd5ea98ed3ad1f3b7e8f24f0b1e6", "bytes": 1049615648,
        "mmproj": ("mmproj-LFM3-VL-2B-Q8_0.gguf",
                   "c5c5b58d544a18da919172d2d0368090ea1a4a351b6feb7f8a326a264a5918ab", 203031872),
        # The model card's recommended sampler.
        "sampler": ["--temp", "0.1", "--top-k", "50", "--top-p", "1.0", "--min-p", "0", "--repeat-penalty", "1.1"],
        "note": "Liquid AI's 2B vision model, quantization-aware Q4_0. General purpose; not tuned for Home Assistant.",
        # Pi 5 sweeps: f16 KV + ubatch 128 read prompts ~5% faster than q8_0 + 512 (93 vs 88 tok/s), and flash
        # attention off made HA-like turns 12% faster (generation 8.9 -> 10.3 tok/s).
        "flash_attn": "off",
        "defaults": {**COMMON_PARAMS, "ubatch": 128, "kv_type": "f16"},
    },
    "bonsai-2-2b": {
        "label": "Bonsai 2 2B", "engine": "ik", "size": "2B",
        "note": "Ternary 2B, the fastest writer. Q2_0 ~13.7 tok/s generation; PQ2_0 reads prompts ~1.7x faster.",
        "defaults": {**COMMON_PARAMS, **SPEC_OFF, "format": "Q2_0"},
    },
    "millie-1.1": {
        "label": "Millie 1.1 35B-A3B", "engine": "millie",
        "repo": "llmsforall/Millie-1.1-35B-A3B-Ternary", "file": "Millie-1.1-35B-A3B-Ternary.gguf",
        "sha256": "9758b69fa24715580249c4879a20bbbe8eb5fff3c94c8cce7110d20eea195db9", "bytes": 8310684160,
        "note": "35B MoE, 8.3 GB: needs an NVMe drive and the 8 GB Pi. Cold experts stream from disk; ~4.6 tok/s.",
        "defaults": {**COMMON_PARAMS, "threads": 3, "ubatch": 512, "hot_experts": 32, "hot_bias": "1", "map_dense": False},
    },
}

PARAM_SPECS = {
    "threads": {"label": "Threads", "type": "int", "min": 1, "max": 4},
    "ctx_size": {"label": "Context", "type": "int", "min": 512, "max": 32768, "step": 512},
    "ubatch": {"label": "Prompt micro-batch", "type": "int", "min": 32, "max": 512, "step": 32},
    "kv_type": {"label": "KV cache", "type": "choice", "choices": ["q8_0", "f16"]},
    "format": {"label": "File format", "type": "choice", "choices": ["PQ2_0", "Q2_0"],
               "help": "PQ2_0 is repacked into RAM (fast prompts); Q2_0 is memory-mapped (fast generation on 2B/4B)."},
    "hot_experts": {"label": "Hot experts / layer", "type": "int", "min": 8, "max": 128, "step": 8,
                    "help": "Resident experts; each +16 costs ~0.5 GB. Cold ones are read from NVMe."},
    "vision": {"label": "Vision (images)", "type": "bool",
               "help": "Loads the image encoder (+0.3-0.6 GB RAM). An image takes ~30-40 s before the first word."},
    "image_tokens": {"label": "Image tokens (max)", "type": "int", "min": 64, "max": 1024, "step": 64,
                     "help": "Detail per image. Encode time grows fast above 256."},
    "speculative": {"label": "Speculative", "type": "choice", "choices": list(SPECULATIVE),
                    "help": "Drafts tokens from the context and verifies them in one batch. Text-only for now."},
    "hot_bias": {"label": "Hot-expert routing bias", "type": "choice", "choices": ["1", "0"],
                 "help": "1 prefers experts already in RAM (faster, changes routing); 0 routes exactly as the model does."},
    "map_dense": {"label": "Map dense weights", "type": "bool",
                  "help": "Map ~1.4 GB of non-expert weights read-only (less RAM, no repack)."},
}


def model_file(model_id, params):
    m = MODELS[model_id]
    if m["engine"] in ("millie", "llama"):
        return m["repo"], m["file"], m["sha256"], m["bytes"]
    fmt = params.get("format", "PQ2_0")
    if "files" in m:
        sha, size = m["files"][fmt]
        return None, f"{m['file_prefix']}-{fmt}.gguf", sha, size
    sha, size = BONSAI_FILES[m["size"]][fmt]
    return (f"prism-ml/Ternary-Bonsai-2-{m['size']}-gguf", f"Ternary-Bonsai-2-{m['size']}-{fmt}.gguf", sha, size)


def mmproj_file(model_id):
    """(repo, file, sha256, bytes) of the model's vision projector."""
    m = MODELS[model_id]
    if m["engine"] == "millie":
        return (m["repo"], *MILLIE_MMPROJ)
    if "mmproj" in m:
        return (m.get("mmproj_repo", m["repo"]), *m["mmproj"])
    if m.get("text_only"):
        return (m["repo"], None, None, 0)
    sha, size = BONSAI_MMPROJ[m["size"]]
    return (f"prism-ml/Ternary-Bonsai-2-{m['size']}-gguf", f"Ternary-Bonsai-2-{m['size']}-mmproj-Q8_0.gguf", sha, size)


def engine_args(model_id, params, path, port):
    m = MODELS[model_id]
    p = params
    if m.get("flash_attn") == "off":
        p = {**p, "kv_type": "f16"}  # a quantized V cache needs flash attention
    args = [ENGINES[m["engine"]]["server"], "--model", str(path), "--alias", model_id,
            "--host", ENGINE_HOST, "--port", str(port),
            "--threads", str(p["threads"]), "--ctx-size", str(p["ctx_size"]), "--parallel", "1",
            "--ubatch-size", str(p["ubatch"]), "--cache-type-k", p["kv_type"], "--cache-type-v", p["kv_type"],
            "--flash-attn", m.get("flash_attn", "on"), "--jinja", "--no-warmup"]
    if p.get("vision"):
        args += ["--mmproj", str(MODEL_DIR / mmproj_file(model_id)[1]), "--image-max-tokens", str(p["image_tokens"])]
    if m["engine"] == "ik":
        args += ["--cache-ram", "1024", "--reasoning-format", "none",
                 "--chat-template-kwargs", '{"enable_thinking": false}']
        if p.get("format") == "PQ2_0":
            args.append("-rtr")
        if SPECULATIVE.get(p.get("speculative")):
            args += ["--spec-type", SPECULATIVE[p["speculative"]]]
        args += m.get("sampler", [])
    elif m["engine"] == "llama":
        # Same prompt cache as ik; reasoning off opens each turn with an empty think block where the template has one.
        # --swa-full: LFM3's attention has a 256-token sliding window, and with it the server only resumes from a
        # checkpoint ~256 tokens before the new text, so every HA request re-read ~515 tokens. The full-size cache
        # lets it resume where the user message starts (a few tens of MB more at 8k context).
        args += ["--cache-ram", "1024", "--swa-full", "--reasoning", "off", "--reasoning-format", "none"]
        if (SHARE / "debug-engine").exists():  # trace level: checkpoint creation, restore and eviction
            args += ["--log-verbosity", "4"]
        args += m.get("sampler", [])
    else:
        # preserve_thinking keeps the (empty) think block on earlier replies, as Millie saw them when it wrote them:
        # without it a follow-up diverges right after the first reply starts, and the hybrid model re-reads everything.
        args += ["--no-mmap", "--hot-experts", str(p["hot_experts"]), "--hot-expert-bias", p.get("hot_bias", "1"),
                 "--reasoning", "off",
                 "--chat-template-kwargs", json.dumps(m.get("template_kwargs", {"preserve_thinking": True}))]
        if m.get("template"):
            args += ["--chat-template-file", str(MODEL_DIR / m["template"])]
        if p.get("map_dense"):
            args.append("--hot-map-dense")
        if m.get("lora"):
            args += ["--lora", str(MODEL_DIR / m["lora"])]
        args += m.get("sampler", [])
    return args


def clean_params(model_id, params):
    out = copy.deepcopy(MODELS[model_id]["defaults"])
    for key, value in (params or {}).items():
        if key not in out:
            continue
        spec = PARAM_SPECS[key]
        if spec["type"] == "int":
            value = max(spec["min"], min(spec["max"], int(value)))
        elif spec["type"] == "choice" and value not in spec["choices"]:
            continue
        elif spec["type"] == "bool":
            value = bool(value)
        out[key] = value
    if MODELS[model_id].get("text_only"):
        out["vision"] = False  # no projector to load
    return out


# ---------------------------------------------------------------- firmware mailbox (vcgencmd)

IOCTL_MBOX_PROPERTY = 0xC0086400  # _IOWR(100, 0, char *) on aarch64
GENCMD_TAG = 0x00030080
GENCMD_MAX = 4096


def gencmd(command):
    """Run a vcgencmd command through the firmware mailbox; None if /dev/vcio is unavailable."""
    for dev in ("/dev/vcio_gencmd", "/dev/vcio"):
        try:
            fd = os.open(dev, os.O_RDONLY)
        except OSError:
            continue
        try:
            # size, request, tag, buffer length, request length, error, command[GENCMD_MAX], end tag
            size = 6 * 4 + GENCMD_MAX + 4
            buf = bytearray(size)
            struct.pack_into("<6I", buf, 0, size, 0, GENCMD_TAG, GENCMD_MAX, 0, 0)
            raw = command.encode()[:GENCMD_MAX - 1]
            buf[24:24 + len(raw)] = raw
            fcntl.ioctl(fd, IOCTL_MBOX_PROPERTY, buf, True)
            return bytes(buf[24:24 + GENCMD_MAX]).split(b"\0", 1)[0].decode(errors="replace").strip()
        except OSError:
            return None
        finally:
            os.close(fd)
    return None


THROTTLE_BITS = {0: "under-voltage", 1: "freq capped", 2: "throttled", 3: "soft temp limit"}


def firmware_snapshot():
    snap = {}
    t = gencmd("get_throttled")
    if t and "=" in t:
        value = int(t.split("=", 1)[1], 16)
        snap["throttled_raw"] = hex(value)
        snap["throttled_now"] = [name for bit, name in THROTTLE_BITS.items() if value & (1 << bit)]
        snap["throttled_since_boot"] = [name for bit, name in THROTTLE_BITS.items() if value & (1 << (bit + 16))]
    c = gencmd("measure_clock arm")
    if c and "=" in c:
        snap["arm_mhz_measured"] = round(int(c.split("=", 1)[1]) / 1e6)
    v = gencmd("measure_volts core")
    if v and "=" in v:
        snap["core_volts"] = float(v.split("=", 1)[1].rstrip("V"))
    adc = gencmd("pmic_read_adc")
    if adc:
        amps, volts = {}, {}
        for name, kind, val in re.findall(r"(\S+?)_[AV]\s+(current|volt)\(\d+\)=([\d.]+)", adc):
            (amps if kind == "current" else volts)[name] = float(val)
        rails = sum(amps[n] * volts[n] for n in amps if n in volts)
        if rails:
            snap["pmic_rails_watts"] = round(rails, 2)
        ext = re.search(r"EXT5V_V volt\(\d+\)=([\d.]+)", adc)
        if ext:
            snap["ext5v_volts"] = float(ext.group(1))
    return snap


# ---------------------------------------------------------------- telemetry

def read(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def hwmon():
    out = {}
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        name = read(h / "name", "")
        if name == "cpu_thermal":
            out["cpu_temp_c"] = int(read(h / "temp1_input", 0)) / 1000
        elif name == "nvme":
            out["nvme_temp_c"] = int(read(h / "temp1_input", 0)) / 1000
        elif name == "rp1_adc":
            out["rp1_temp_c"] = int(read(h / "temp1_input", 0)) / 1000
        elif name == "pwmfan":
            out["fan_rpm"] = int(read(h / "fan1_input", 0) or 0)
            out["fan_pwm_pct"] = round(int(read(h / "pwm1", 0) or 0) / 255 * 100)
        elif name == "rpi_volt":
            out["undervoltage_alarm"] = read(h / "in0_lcrit_alarm", "0") == "1"
    return out


def app_memory():
    """The whole app container (controller + engine + its page cache), from its cgroup. 'used' matches the
    Supervisor's/Docker's memory_usage (current minus inactive file cache)."""
    current = read("/sys/fs/cgroup/memory.current")
    if not current:
        return {}
    stat = {}
    for line in read("/sys/fs/cgroup/memory.stat", "").splitlines():
        key, _, value = line.partition(" ")
        stat[key] = int(value or 0)
    mib = lambda b: b // 2**20
    own = {l.split(":")[0]: int(l.split()[1]) for l in read("/proc/self/status", "").splitlines()
           if l.startswith(("VmRSS", "VmHWM"))}
    return {"app_used_mib": mib(int(current) - stat.get("inactive_file", 0)), "app_total_mib": mib(int(current)),
            "app_anon_mib": mib(stat.get("anon", 0)), "app_file_mib": mib(stat.get("file", 0)),
            "controller_rss_mib": own.get("VmRSS", 0) // 1024}


class Telemetry:
    def __init__(self, engine):
        self.engine = engine
        self.latest = {}
        self.history = deque(maxlen=300)  # 10 min at 2 s
        self.max_temp = 0.0
        self._cpu_prev = None
        self._fw_at = 0.0
        self._fw = {}

    def _cpu_usage(self):
        lines = [l.split() for l in read("/proc/stat", "").splitlines() if l.startswith("cpu")]
        now = {l[0]: (sum(int(x) for x in l[1:]), int(l[4]) + int(l[5])) for l in lines}
        usage = {}
        if self._cpu_prev:
            for k, (total, idle) in now.items():
                pt, pi = self._cpu_prev.get(k, (total, idle))
                dt = total - pt
                usage[k] = round(100 * (1 - (idle - pi) / dt)) if dt else 0
        self._cpu_prev = now
        return usage

    def sample(self):
        s = {"time": time.time()}
        s.update(hwmon())
        usage = self._cpu_usage()
        s["cpu_pct"] = usage.get("cpu", 0)
        s["cores_pct"] = [usage.get(f"cpu{i}", 0) for i in range(os.cpu_count() or 4)]
        s["cores_mhz"] = [int(read(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq", 0) or 0) // 1000
                          for i in range(os.cpu_count() or 4)]
        s["load"] = [float(x) for x in read("/proc/loadavg", "0 0 0").split()[:3]]
        mem = {l.split(":")[0]: int(l.split()[1]) for l in read("/proc/meminfo", "").splitlines() if ":" in l}
        s["mem_total_mib"] = mem.get("MemTotal", 0) // 1024
        s["mem_available_mib"] = mem.get("MemAvailable", 0) // 1024
        s["swap_used_mib"] = (mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) // 1024
        psi = read("/proc/pressure/memory", "")
        m = re.search(r"some avg10=([\d.]+)", psi or "")
        s["mem_pressure_some10"] = float(m.group(1)) if m else None
        s.update(self.engine.process_memory())
        s.update(app_memory())
        disk = shutil.disk_usage(MODEL_DIR if MODEL_DIR.exists() else "/")
        s["share_free_gib"] = round(disk.free / 2**30, 1)
        if time.time() - self._fw_at > 4:
            self._fw = firmware_snapshot()
            self._fw_at = time.time()
        s["firmware"] = self._fw
        if s.get("cpu_temp_c"):
            self.max_temp = max(self.max_temp, s["cpu_temp_c"])
        s["max_temp_c"] = self.max_temp
        self.latest = s
        self.history.append({k: s.get(k) for k in ("time", "cpu_temp_c", "cpu_pct", "mem_available_mib", "fan_rpm",
                                                   "app_used_mib")}
                            | {"arm_mhz": self._fw.get("arm_mhz_measured")})

    async def run(self):
        while True:
            try:
                await asyncio.get_running_loop().run_in_executor(None, self.sample)
            except Exception:  # telemetry must never take the controller down
                log.exception("telemetry sample failed")
            await asyncio.sleep(2)


def platform_info():
    info = {"model": read("/proc/device-tree/model", "").rstrip("\0"), "kernel": os.uname().release}
    if not info["model"]:  # the device tree isn't mounted in app containers; the Pi kernel repeats it here
        m = re.search(r"^Model\s*:\s*(.+)$", read("/proc/cpuinfo", ""), re.M)
        info["model"] = m.group(1).strip() if m else ""
    for line in read("/etc/os-release", "").splitlines():
        if line.startswith("PRETTY_NAME="):
            info["container_os"] = line.split("=", 1)[1].strip('"')
    fw = gencmd("version")
    if fw:
        info["firmware"] = fw.splitlines()[0]
    boot = gencmd("bootloader_version")
    if boot:
        info["bootloader"] = boot.splitlines()[0]
    return info


# ---------------------------------------------------------------- engine manager

class Engine:
    def __init__(self):
        self.state = "stopped"  # stopped | downloading | loading | ready | error
        self.model = None
        self.params = {}
        self.message = ""
        self.download = None  # {"file", "done", "total"}
        self.loaded_at = None
        self.load_seconds = None
        self.proc = None
        self.url = None
        self.model_dir = MODEL_DIR
        self.logs = deque(maxlen=200)
        self._file_locks = {}
        self._task = None
        self._lock = asyncio.Lock()
        self.on_ready = []  # coroutine functions run (in the background) each time an engine starts serving
        self._ready_tasks = []
        self._jobs = {}  # work the UI didn't ask for (HA prefix warm-up, HA requests) -> (what, started)

    @contextlib.contextmanager
    def busy_with(self, what):
        """Mark the engine busy with background work. It serves one request at a time, so a chat sent from
        the UI meanwhile waits; the UI shows why."""
        key = object()
        self._jobs[key] = (what, time.time())
        try:
            yield
        finally:
            self._jobs.pop(key, None)

    def status(self):
        now = time.time()
        return {"state": self.state, "model": self.model, "params": self.params, "message": self.message,
                "download": self.download, "load_seconds": self.load_seconds,
                "busy": [{"what": what, "seconds": round(now - started)} for what, started in self._jobs.values()],
                "engine": MODELS[self.model]["engine"] if self.model else None}

    def process_memory(self):
        if not self.proc or self.proc.returncode is not None:
            return {}
        fields = {}
        for line in read(f"/proc/{self.proc.pid}/status", "").splitlines():
            key = line.split(":")[0]
            if key in ("VmRSS", "RssAnon", "RssFile", "VmHWM"):
                fields[key] = int(line.split()[1]) // 1024
        return {"engine_rss_mib": fields.get("VmRSS"), "engine_anon_mib": fields.get("RssAnon"),
                "engine_file_mib": fields.get("RssFile"), "engine_peak_mib": fields.get("VmHWM")}

    async def load(self, model_id, params):
        async with self._lock:
            await self.unload()
            self.model, self.params = model_id, clean_params(model_id, params)
            self.state, self.message = "loading", "starting"
            self._task = asyncio.create_task(self._run())

    async def unload(self):
        """Cancel any download or load in progress and stop the engine."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        await self.stop()
        self.download = None

    async def stop(self):
        if self.proc and self.proc.returncode is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self.proc.wait(), 20)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        self.proc = None
        self.state = "stopped"
        self.message = ""

    async def _run(self):
        wanted = [model_file(self.model, self.params)]
        if self.params.get("vision"):
            wanted.append(mmproj_file(self.model))
        path = MODEL_DIR / wanted[0][1]
        try:
            if not all((MODEL_DIR / f[1]).exists() for f in wanted):
                self.state = "downloading"
                await self.fetch_files(wanted)
            self.state, self.message = "loading", "starting engine"
            started = time.time()
            port = free_port()
            self.url = f"http://{ENGINE_HOST}:{port}"
            args = engine_args(self.model, self.params, path, port)
            self.logs.append("$ " + " ".join(args))
            self.proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            reader = asyncio.create_task(self._pump_logs(self.proc))
            async with aiohttp.ClientSession() as session:
                while True:
                    if self.proc.returncode is not None:
                        raise RuntimeError(f"engine exited ({self.proc.returncode}): " + " | ".join(list(self.logs)[-3:]))
                    with contextlib.suppress(aiohttp.ClientError, asyncio.TimeoutError):
                        async with session.get(f"{self.url}/health", timeout=aiohttp.ClientTimeout(total=2)) as r:
                            if r.status == 200:
                                break
                    await asyncio.sleep(1)
            self.load_seconds = round(time.time() - started, 1)
            self.loaded_at = time.time()
            self.state, self.message = "ready", ""
            self._ready_tasks = [asyncio.create_task(cb()) for cb in self.on_ready]
            await reader
            if self.state == "ready":
                self.state, self.message = "error", "engine exited: " + " | ".join(list(self.logs)[-3:])
        except asyncio.CancelledError:
            if "reader" in locals():
                reader.cancel()
            raise
        except Exception as err:
            log.exception("engine run failed")
            self.state, self.message = "error", str(err)[:600]

    async def _pump_logs(self, proc):
        # With debug-engine in the share, keep the whole engine log too: trace output outruns the 200-line buffer.
        debug = open(SHARE / "engine-debug.log", "w") if (SHARE / "debug-engine").exists() else None
        while True:
            line = await proc.stdout.readline()
            if not line:
                await proc.wait()
                if debug:
                    debug.close()
                return
            text = line.decode(errors="replace").rstrip()
            self.logs.append(text)
            if debug:
                debug.write(text + "\n")
                debug.flush()
            if self.state == "loading" and text:
                self.message = text[-160:]

    async def fetch_files(self, files):
        """Download any missing (repo, file, sha256, bytes) entries; also used by the Ollama /api/pull."""
        for repo, file, sha, size in files:
            path = MODEL_DIR / file
            async with self._file_locks.setdefault(file, asyncio.Lock()):
                if not path.exists():
                    await self._download(repo, file, sha, size, path)

    async def _download(self, repo, file, sha, size, path):
        if repo is None:
            raise RuntimeError(f"{file} is a local model; copy it into {MODEL_DIR}")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        self.download = {"file": file, "done": part.stat().st_size if part.exists() else 0, "total": size}
        url = f"https://huggingface.co/{repo}/resolve/main/{file}"
        headers = {"Range": f"bytes={self.download['done']}-"} if self.download["done"] else {}
        timeout = aiohttp.ClientTimeout(total=None, sock_read=120)
        async with aiohttp.ClientSession(timeout=timeout) as session, session.get(url, headers=headers) as r:
            if r.status not in (200, 206):
                raise RuntimeError(f"download failed: HTTP {r.status}")
            if r.status == 200:
                self.download["done"] = 0
            with open(part, "ab" if r.status == 206 else "wb") as fh:
                async for chunk in r.content.iter_chunked(1 << 20):
                    fh.write(chunk)
                    self.download["done"] += len(chunk)
        self.message = "verifying SHA-256"
        digest = await asyncio.get_running_loop().run_in_executor(None, sha256_file, part)
        if digest != sha:
            part.unlink()
            raise RuntimeError(f"SHA-256 mismatch for {file}; the download was removed")
        part.rename(path)
        self.download = None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- settings

def load_settings():
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2))
    tmp.replace(SETTINGS_FILE)


def default_model():
    try:
        opt = json.loads(OPTIONS_FILE.read_text()).get("default_model")
    except (OSError, ValueError):
        opt = None
    return opt if opt in MODELS else "lfm3-1.7b-home-assistant"


# ---------------------------------------------------------------- HTTP

async def index(request):
    return web.FileResponse(WWW / "index.html")


async def api_status(request):
    engine = request.app["engine"]
    settings = request.app["settings"]
    models = {}
    for mid, m in MODELS.items():
        params = clean_params(mid, settings.get("params", {}).get(mid))
        _, file, _, size = model_file(mid, params)
        _, vfile, _, vsize = mmproj_file(mid)
        downloaded = (MODEL_DIR / file).exists() and (not params.get("vision") or (MODEL_DIR / vfile).exists())
        models[mid] = {"label": m["label"], "engine": m["engine"], "note": m["note"], "defaults": m["defaults"],
                       "params": params, "file": file, "bytes": size + (vsize if params.get("vision") else 0),
                       "downloaded": downloaded}
    return web.json_response({"engine": engine.status(), "models": models, "param_specs": PARAM_SPECS,
                              "engines": {k: v["label"] for k, v in ENGINES.items()},
                              "platform": request.app["platform"]})


async def api_telemetry(request):
    t = request.app["telemetry"]
    return web.json_response({"latest": t.latest, "history": list(t.history)[-90:]})


async def api_load(request):
    body = await request.json()
    model_id = body.get("model")
    if model_id not in MODELS:
        raise web.HTTPBadRequest(text="unknown model")
    params = clean_params(model_id, body.get("params"))
    settings = request.app["settings"]
    settings["model"] = model_id
    settings.setdefault("params", {})[model_id] = params
    save_settings(settings)
    await request.app["engine"].load(model_id, params)
    return web.json_response({"ok": True, "params": params})


async def api_stop(request):
    await request.app["engine"].unload()
    return web.json_response({"ok": True})


async def api_logs(request):
    return web.Response(text="\n".join(request.app["engine"].logs))


async def proxy(request, path=None):
    engine = request.app["engine"]
    if engine.state != "ready":
        return web.json_response({"error": {"message": f"model not ready ({engine.state})"}}, status=503)
    url = f"{engine.url}/{path or request.match_info['tail']}"
    body = await request.read()
    headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "accept", "authorization")}
    session = request.app["client"]
    async with session.request(request.method, url, params=request.query, data=body, headers=headers) as upstream:
        resp = web.StreamResponse(status=upstream.status)
        for key in ("Content-Type", "Cache-Control"):
            if key in upstream.headers:
                resp.headers[key] = upstream.headers[key]
        await resp.prepare(request)
        try:
            async for chunk in upstream.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
        except (ConnectionResetError, aiohttp.ClientConnectionResetError):
            pass  # the client went away (Stop, closed tab); leaving the block closes the engine request too
        return resp


async def health(request):
    engine = request.app["engine"]
    return web.json_response({"status": engine.state}, status=200 if engine.state == "ready" else 503)


async def on_startup(app):
    app["client"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=900))
    app["platform"] = await asyncio.get_running_loop().run_in_executor(None, platform_info)
    app["telemetry_task"] = asyncio.create_task(app["telemetry"].run())
    settings = app["settings"]
    model_id = settings.get("model") if settings.get("model") in MODELS else default_model()
    await app["engine"].load(model_id, settings.get("params", {}).get(model_id))


async def on_cleanup(app):
    app["telemetry_task"].cancel()
    await app["engine"].unload()
    await app["client"].close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = Engine()
    app = web.Application(client_max_size=64 * 2**20)  # room for base64 images in chat requests
    app["engine"] = engine
    app["telemetry"] = Telemetry(engine)
    app["settings"] = load_settings()
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/telemetry", api_telemetry)
    app.router.add_get("/api/logs", api_logs)
    app.router.add_post("/api/load", api_load)
    app.router.add_post("/api/stop", api_stop)
    app.router.add_route("*", "/v1/{tail:.*}", proxy_v1)
    OllamaAPI(app, MODELS, engine, model_file, mmproj_file, clean_params,
              lambda mid: clean_params(mid, app["settings"].get("params", {}).get(mid))).register(app.router)
    app.router.add_route("*", "/{tail:(?:completion|completions|tokenize|detokenize|apply-template|props|slots).*}", proxy)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, host="0.0.0.0", port=LISTEN_PORT, access_log=None)


async def proxy_v1(request):
    return await proxy(request, "v1/" + request.match_info["tail"])


if __name__ == "__main__":
    main()
