# llm-tunnel

A tiny repo-local launcher for [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` on macOS Apple Silicon, with optional Cloudflare Tunnel sharing.

Everything mutable lives inside this repo: binaries in `bin/`, models in `models/`, logs in `logs/`, PID files in `run/`, secrets in `config.env`. No Homebrew, no Docker, no Python, no LaunchAgents.

## Layout

```
llm                  the only script. up | down | status
config.example.env   committed template
config.env           your local config (gitignored, holds secrets)
bin/                 llama-server, cloudflared (gitignored)
models/              your GGUF (gitignored)
logs/                process logs (gitignored)
run/                 PID files (gitignored)
```

## Install (once)

1. Get the binaries. Pick the latest macOS arm64 builds from each project's GitHub releases.

   ```sh
   # llama-server: https://github.com/ggml-org/llama.cpp/releases (look for llama-bXXXX-bin-macos-arm64.zip)
   # unzip, copy the llama-server binary into ./bin/
   chmod +x ./bin/llama-server

   # cloudflared: https://github.com/cloudflare/cloudflared/releases (cloudflared-darwin-arm64.tgz)
   # untar, copy the cloudflared binary into ./bin/
   chmod +x ./bin/cloudflared
   ```

   On first run macOS may quarantine the binaries — right-click → Open once, or `xattr -d com.apple.quarantine ./bin/llama-server ./bin/cloudflared`.

2. Drop a GGUF model at `./models/model.gguf` (or change `LLM_MODEL` in `config.env`).

3. Copy the config template:

   ```sh
   cp config.example.env config.env
   ```

   Edit `config.env` if needed. It is gitignored, so any secrets you put in it stay local.

## Use

```sh
./llm up       # start (and start cloudflared if a token is set)
./llm status   # see what's running, hit /health
./llm down     # stop everything
```

The server runs in the background with `nohup` — closing the terminal won't kill it. A reboot will, since nothing is installed as a login service. To restart: `./llm down && ./llm up`.

Default URL: <http://127.0.0.1:8080>

## Coding agents

`llama-server` exposes OpenAI-compatible endpoints at `/v1`. For tools that accept a custom base URL:

```sh
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="local"
```

Agent workloads are long-context and memory-heavy — consider `LLAMA_PARALLEL=1` and a bigger `LLAMA_CONTEXT` in `config.env`. Compatibility depends on whether the agent honors `OPENAI_BASE_URL` and whether your model has a sane chat template.

## LAN access

Set `LLM_HOST="0.0.0.0"` in `config.env` and `./llm up` again. Other devices on your network can hit `http://<your-mac-lan-ip>:8080`. Trusted networks only — there's no auth.

Find your LAN IP: `ipconfig getifaddr en0` (or `en1`).

## Cloudflare share (named tunnel)

For a stable URL friends can use:

1. In the Cloudflare Zero Trust dashboard, create a tunnel and copy the connector token.
2. Paste it into `config.env`:

   ```sh
   CLOUDFLARE_TUNNEL_TOKEN="eyJ..."
   ```

3. In the tunnel's "Public Hostnames" config, point it at `http://localhost:8080`.
4. `./llm up` — `cloudflared` starts alongside `llama-server`.

`config.env` is gitignored, so the token never ends up in git.

## Temporary URL (Quick Tunnel)

If you just want a random throwaway URL right now:

```sh
./bin/cloudflared tunnel --url http://127.0.0.1:8080
```

This runs in the foreground and prints a `*.trycloudflare.com` URL. Random per-run, not for long-term use.

## Logs

```sh
tail -f logs/*.log
```

## Privacy

Prompts are processed on this Mac by `llama-server`. There is no database and no chat history; the only on-disk state is process logs in `logs/` (which don't include request bodies unless `llama-server` is started with debug flags). When you expose the server via LAN or Cloudflare, anyone who can reach the URL can use the model — there's no auth at the `llama-server` layer.

Reasonable thing to tell a friend: *"Your prompts are processed on my Mac. No chat history is saved. The shared link is gated by Cloudflare."*
