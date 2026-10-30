# Potato Assist

A local LLM for Home Assistant's Assist, running on your Raspberry Pi 5 as a Home Assistant app. It plugs into Home
Assistant's built-in Ollama integration, so Assist can answer questions about your home and control devices without
a cloud service or a GPU. From [Potato OS](https://github.com/potato-os/core).

[![Add this repository to your Home Assistant](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fpotato-os%2Fpotato-assist)

## What you get

- **[lfm3-1.7b-home-assistant](https://huggingface.co/Slomin/lfm3-1.7b-home-assistant)**, a 1 GB model fine-tuned
  for Home Assistant tool calls, as the default. It scores 92.5 % on the
  [Home LLM Leaderboard](https://github.com/allenporter/home-assistant-datasets/tree/main/reports) datasets, against
  67.8 % for its base model.
- About **6 seconds** per Assist request on a Pi 5, with Home Assistant's prompt cached between requests.
- An **Ollama-compatible API** for Home Assistant's Ollama integration, an OpenAI-compatible `/v1`, and a chat UI
  with image input in the Home Assistant sidebar.
- **Pi 5 telemetry** in the UI: CPU temperature, throttling, measured clock and power.
- Other models to try: Liquid AI's LFM3-VL 2B, Bonsai 2 2B (ternary) and Millie 1.1 35B-A3B (streams from NVMe).

## On a real home

On a Raspberry Pi 5 (8 GB) with Home Assistant 2026.10, 22 real requests run twice through Assist:

| Model | Correct | Mean time | Unwanted device actions |
|---|---|---|---|
| lfm3-1.7b-home-assistant v0.1 | 42/44 | 6.0 s | 0 |
| LFM3-VL 2B (base model) | 37/44 | 7.2 s | 4 |

The requests cover sensor readings, lights (including ones that are offline), the shopping list, the time, and requests
that should be refused, such as devices that aren't exposed.

## Install

1. Click the button above, or add `https://github.com/potato-os/potato-assist` under **Settings → Apps → App store →
   ⋮ → Repositories**.
2. Install **Potato Assist** and start it.
3. Follow the app's **Documentation** tab to connect the Ollama integration.

Requirements: a Raspberry Pi 5 running Home Assistant OS. 8 GB of RAM is recommended.

## License

Apache-2.0. The engines are [llama.cpp](https://github.com/ggml-org/llama.cpp),
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and
[millie-runtime](https://github.com/llmsforall/millie-runtime) (MIT), with the patches in `potato_assist/`. Models are
downloaded from Hugging Face under their own licenses.
