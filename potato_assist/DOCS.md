# Potato Assist

Potato Assist runs a small language model on your Raspberry Pi 5, next to Home Assistant, and connects it to Assist
through Home Assistant's built-in **Ollama** integration. Nothing leaves your network except the one-time model
download from Hugging Face.

## Requirements

- Raspberry Pi 5 running Home Assistant OS. The engines are built for the Pi 5's CPU (Cortex-A76), not for a Pi 4.
- Tested on the 8 GB Pi 5. The 2B models use about 2.5 GB of RAM.
- About 1.3 GB of free disk for the default model. Millie 1.1 needs 8.5 GB and an NVMe drive.

## Setup

1. Install and start the app. On first start it downloads the default model (about 1.3 GB). Open the app's web UI
   to follow the download and to chat with the model.
2. Copy the app's **Hostname** from its Info page (it looks like `xxxxxxxx-potato-assist`).
3. In Home Assistant, go to **Settings → Devices & services → Add integration → Ollama** and enter
   `http://<hostname>:8080` as the URL.
4. Add a conversation agent, pick the model `lfm3-1.7b-home-assistant`, and turn on **Assist** control.
5. In **Settings → Voice assistants**, select the agent for your assistant. Keep **Prefer handling commands locally**
   on: Home Assistant then handles simple commands instantly, and the model answers everything else.

## Models

| Model | Size | Notes |
|---|---|---|
| `lfm3-1.7b-home-assistant` (default) | 1.05 GB + 0.2 GB vision | Fine-tuned for Home Assistant. About 6 s per Assist request. |
| `lfm3-vl-2b` | 1.05 GB + 0.2 GB vision | Liquid AI's base model, general purpose. |
| `bonsai-2-2b` | 0.55 GB + 0.36 GB vision | Ternary 2B, fastest at writing long answers. |
| `millie-1.1` | 8.3 GB + 0.3 GB vision | 35B mixture of experts that streams from disk. Slow (about 4.6 tokens/s), but knows the most. |

Switch models in the app's web UI. Only one model is loaded at a time. Models are stored in
`/share/potato-assist/models`.

Tips for good results with a small model:

- Expose only the entities you want Assist to control (**Settings → Voice assistants → Expose**). Fewer entities
  mean faster and more accurate answers.
- Put devices in areas. The model sees the home grouped by area.
- The first request after Home Assistant's prompt changes (new exposed entities, new instructions) takes about 30 s;
  later requests take a few seconds.

## Network port

Home Assistant talks to the app over the internal network, so port 8080 is closed by default. Opening it in the
app's **Network** settings makes the chat UI, an OpenAI-compatible API (`/v1`) and an Ollama-compatible API (`/api`)
available to your whole LAN **without a password**.

## Troubleshooting

- **Logs:** the app's Log tab shows engine output and download progress.
- **Engine trace:** create an empty file `debug-engine` in `/share/potato-assist`. After the next model load, the
  engine log is written in full to `/share/potato-assist/engine-debug.log`.
- **Last request:** create an empty file `debug-ollama` in `/share/potato-assist`. Each request from Home Assistant
  is then saved to `/share/potato-assist/last-ollama-request.json`. It contains your home's exposed entities and
  states, so delete both files afterwards.

## Licenses

The app is Apache-2.0. The engines are llama.cpp, ik_llama.cpp and millie-runtime (all MIT). Each model has its own
license on its Hugging Face page; the LFM3 models use the LFM Open License v1.0, which limits commercial use to
companies with under US$10 million annual revenue.
