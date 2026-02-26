"""
EB-ALFRED Remote Environment Server.

Starts a FastAPI service that exposes EB-ALFRED as a remote gym environment.
The service can run on a machine with GPU + X server (for AI2-THOR rendering),
while VAGEN RL training runs on a separate machine using GymImageEnvClient.

Usage:
    python -m vagen.envs.eb_alfred.serve --port 8000 --x-display 1

    # Then on the training machine, configure env_config:
    #   base_urls: ["http://<server-ip>:8000"]
    #   eval_set: "base"
    #   resolution: 500
"""

import argparse
import uvicorn

from vagen.envs_remote.service import build_gym_service
from .handler import EbAlfredHandler


def main():
    parser = argparse.ArgumentParser(description="EB-ALFRED Remote Environment Server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--session-timeout",
        type=float,
        default=3600.0,
        help="Session timeout in seconds",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Max concurrent sessions (0=unlimited)",
    )
    args = parser.parse_args()

    handler = EbAlfredHandler(
        session_timeout=args.session_timeout,
        max_sessions=args.max_sessions,
    )
    app = build_gym_service(handler)

    print(f"Starting EB-ALFRED service on {args.host}:{args.port}")
    print(f"Health check: http://localhost:{args.port}/health")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
