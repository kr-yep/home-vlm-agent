# home-vlm-agent

Read `C:/Users/Kris/Documents/DEV/AGENTS.md` first. README.md owns goal, how to run, status and decisions; keep it current instead of adding new docs.

- Everything runs in Docker in WSL (`compose.yaml`); install nothing on the Windows host.
- The model and app containers sit on an internal network with no egress; only `proxy` publishes `127.0.0.1:8700`.
- Weights and media live in `/root/.cache/home-vlm` inside WSL, never in Git.
- Qwen models only after the DEV notice QWEN-EGRESS-20261101 is closed.
