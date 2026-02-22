from __future__ import annotations

from pathlib import Path


_MAP_DIR = Path(__file__).resolve().parent


def list_maps() -> list[str]:
    names: list[str] = []
    for path in sorted(_MAP_DIR.glob("*.json")):
        names.append(path.stem)
    return names


def get_map_path(name: str) -> Path:
    path = _MAP_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown map '{name}'. Available: {', '.join(list_maps())}")
    return path
