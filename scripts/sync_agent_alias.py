from pathlib import Path

root = Path(__file__).resolve().parents[1]
(root / "Agent.md").write_bytes((root / "AGENTS.md").read_bytes())
print("Agent.md synchronized with AGENTS.md")
