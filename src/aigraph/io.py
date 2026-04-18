from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: str | Path, model: Type[T]) -> list[T]:
    path = Path(path)
    items: list[T] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(model.model_validate_json(line))
    return items


def write_jsonl(path: str | Path, items: Iterable[BaseModel]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json(by_alias=True))
            f.write("\n")


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
