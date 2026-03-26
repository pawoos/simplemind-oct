import json
from pathlib import Path
from typing import Dict, Optional

def _load_labels() -> Dict[str, Dict]:
    # Resolve absolute path relative to this file
    here = Path(__file__).resolve().parent
    path = here / "labels_total.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find labels_total.json at {path}")
    with path.open("r") as f:
        data = json.load(f)
    if "id_to_name" not in data:
        raise ValueError("labels_total.json must contain 'id_to_name'")
    return data

_DATA = _load_labels()

_ID_TO_NAME: Dict[int, str] = {int(k): v for k, v in _DATA["id_to_name"].items()}
_NAME_TO_ID: Dict[str, int] = (
    {k: int(v) for k, v in _DATA.get("name_to_id", {}).items()}
    if "name_to_id" in _DATA
    else {v: int(k) for k, v in _ID_TO_NAME.items()}
)

def canon_id_for(name: str) -> Optional[int]:
    """Return canonical ID for a class name or filename stem."""
    n = name.lower().replace(".nii", "").strip()
    return _NAME_TO_ID.get(n)

def name_for_id(idx: int) -> Optional[str]:
    return _ID_TO_NAME.get(int(idx))

def legend_rows():
    return sorted(_ID_TO_NAME.items())

def all_canon_names_sorted():
    return [v for _, v in legend_rows()]

def all_canon_ids_sorted():
    return sorted(_ID_TO_NAME.keys())

# --- Diagnostic check at import
if __name__ == "__main__":
    print(f"Loaded {len(_ID_TO_NAME)} canonical labels.")
    print(list(_ID_TO_NAME.items())[:5])
