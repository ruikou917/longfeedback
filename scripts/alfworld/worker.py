"""Standalone ALFWorld worker speaking line-delimited JSON on stdin/stdout.

This script deliberately has no ``longfeedback`` imports so it can run inside
a dedicated (possibly older-Python, x86) environment that has TextWorld and
ALFWorld installed. The state-hash computation must stay byte-identical to
``longfeedback.environments.base.state_hash``.

Protocol (one JSON object per line):
  {"op": "list_games", "split": "train"}
  {"op": "reset", "game_id": "...", "split": "...", "seed": 0}
  {"op": "step", "action": "..."}
  {"op": "close"}
Responses: {"ok": true, "result": {...}} or {"ok": false, "error": "..."}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def state_hash(game_id, step_index, observation, admissible_actions, score, done):
    payload = json.dumps(
        {
            "game_id": game_id,
            "step_index": step_index,
            "observation": normalize_text(observation),
            "admissible_actions": sorted(normalize_text(a) for a in admissible_actions),
            "score": round(float(score), 6),
            "done": bool(done),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Worker:
    def __init__(self, data_dir: Path, max_steps: int) -> None:
        self.data_dir = data_dir
        self.max_steps = max_steps
        self.env = None
        self.game_id = ""
        self.goal = ""
        self.step_index = 0
        self.done = False

    def list_games(self, split):
        root = self.data_dir / "json_2.1.1" / split
        if not root.is_dir():
            raise FileNotFoundError(f"ALFWorld split directory not found: {root}")
        games = sorted(str(p.relative_to(self.data_dir)) for p in root.glob("**/*.tw-pddl"))
        return {"games": games}

    def observation_payload(self, feedback, game_state):
        won = bool(getattr(game_state, "won", False))
        done = self.done or won or bool(getattr(game_state, "lost", False))
        self.done = done
        admissible = (
            []
            if done
            else sorted(normalize_text(c) for c in (game_state.admissible_commands or []))
        )
        score = 1.0 if won else 0.0
        return {
            "game_id": self.game_id,
            "goal": self.goal,
            "observation": feedback,
            "admissible_actions": admissible,
            "step_index": self.step_index,
            "done": done,
            "score": score,
            "state_hash": state_hash(
                self.game_id, self.step_index, feedback, admissible, score, done
            ),
        }

    def reset(self, game_id, seed):
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        request_infos = textworld.EnvInfos(
            admissible_commands=True, won=True, lost=True, extras=["gamefile"]
        )
        if self.env is not None:
            self.env.close()
        self.env = textworld.start(
            str(self.data_dir / game_id), request_infos, wrappers=[AlfredDemangler()]
        )
        self.env.seed(seed)
        game_state = self.env.reset()
        self.game_id = game_id
        self.goal = str(game_state.objective or "")
        self.step_index = 0
        self.done = False
        return self.observation_payload(str(game_state.feedback or ""), game_state)

    def step(self, action):
        if self.env is None or self.done:
            raise RuntimeError("reset must be called before step on a live episode")
        game_state, _, done = self.env.step(normalize_text(action))
        self.step_index += 1
        self.done = bool(done) or self.step_index >= self.max_steps
        return self.observation_payload(str(game_state.feedback or ""), game_state)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        return {"closed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    arguments = parser.parse_args()
    worker = Worker(arguments.data_dir, arguments.max_steps)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        op = ""
        try:
            request = json.loads(line)
            op = request["op"]
            if op == "list_games":
                result = worker.list_games(request["split"])
            elif op == "reset":
                result = worker.reset(request["game_id"], int(request.get("seed", 0)))
            elif op == "step":
                result = worker.step(request["action"])
            elif op == "close":
                result = worker.close()
            else:
                raise ValueError(f"unknown op {op!r}")
            sys.stdout.write(json.dumps({"ok": True, "result": result}) + "\n")
        except Exception as error:  # protocol boundary: report, never crash
            sys.stdout.write(json.dumps({"ok": False, "error": str(error)}) + "\n")
        sys.stdout.flush()
        if op == "close":
            break


if __name__ == "__main__":
    main()
