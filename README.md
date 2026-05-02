# llm-tunnel

A tiny repo-local launcher for [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` on macOS Apple Silicon, with optional Cloudflare Tunnel sharing.

Everything mutable lives inside this repo: binaries in `bin/`, models in `models/`, logs in `logs/`, PID files in `run/`, secrets in `config.env`. No Homebrew, no Docker, no Python, no LaunchAgents.

## Layout

```
llm                  the only script: up | down | status
config.env           your local config (gitignored)
bin/                 llama-server, cloudflared (gitignored)
models/              your GGUF model (gitignored)
logs/                process logs (gitignored)
```

## Install (once)

### 1. Build llama-server

Requires Xcode Command Line Tools (`xcode-select --install`).

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)
cp build/bin/llama-server /path/to/llm-tunnel/bin/
```

### 2. Add a model

Drop a GGUF file at `./models/model.gguf` (or change `LLM_MODEL` in `config.env`).

### 3. Copy the config template

```sh
cp config.example.env config.env
```

### 4. (Optional) Build cloudflared

Only needed if you want to share the server over the internet.

```sh
git clone https://github.com/cloudflare/cloudflared
cd cloudflared
make cloudflared
cp cloudflared /path/to/llm-tunnel/bin/
```

On first run macOS may quarantine binaries — clear with:

```sh
xattr -d com.atlanta.quarantine ./bin/llama-server ./bin/cloudflared
```

## Use

```sh
./llm up       # start server (and tunnel if configured)
./llm status   # check what's running
./llm down     # stop everything
```

The server runs in the background. Closing the terminal won't kill it; a reboot will.

Default local URL: `http://127.0.0.1:8080`

**Note:** Large models take 30–60 seconds to load after `./llm up`. If you get errors immediately after starting, wait a minute and try again.

## Sharing over the internet

Put `bin/cloudflared` in place (see Install step 4), then run `./llm up`. Since `CLOUDFLARE_TUNNEL_TOKEN` is empty in `config.env`, it starts a **quick tunnel** and prints a public URL:

```
public: https://broad-wolf-abc123.trycloudflare.com
```

Share that URL. Anyone with it can use your model — **there is no auth**. The URL changes every time you restart.

To share settings with someone:

```
Base URL:  https://broad-wolf-abc123.trycloudflare.com/v1
API key:   local   (any non-empty string works)
```

### LAN access only

Set `LLM_HOST="0.0.0.0"` in `config.env`. Other devices on your network can reach `http://<your-mac-lan-ip>:8080`. No tunnel needed.

### Named tunnel (stable URL, requires a domain)

If you have a domain on Cloudflare: go to **Zero Trust → Networks → Tunnels → Create a tunnel**, copy the token, and paste it into `config.env`:

```sh
CLOUDFLARE_TUNNEL_TOKEN="eyJ..."
```

Then add a public hostname in the dashboard pointing to `localhost:8080`. `./llm up` will start cloudflared with the named tunnel automatically.

To add a login wall so only specific people can access your URL, set up **Cloudflare Access**: go to **Zero Trust → Access → Applications → Add → Self-hosted**, enter your tunnel hostname, and add a policy with the email addresses you want to allow. Visitors will be sent a one-time code to their email before they can get through.

## Coding agents

`llama-server` exposes OpenAI-compatible endpoints at `/v1`:

```sh
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"  # or your tunnel URL
export OPENAI_API_KEY="local"
```

## Logs

```sh
tail -f logs/*.log
```

## Privacy

Prompts are processed locally by `llama-server` on this Mac. No chat history is saved. When you share via LAN or quick tunnel, anyone who can reach the URL can use the model — there is no auth unless you set up Cloudflare Access (requires a domain).