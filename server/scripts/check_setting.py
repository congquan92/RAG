from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any


SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
	sys.path.insert(0, str(SERVER_DIR))


def _default_env_file() -> Path:
	project_env = SERVER_DIR.parent / ".env"
	if project_env.exists():
		return project_env
	return SERVER_DIR / ".env"


def _dotenv_values(path: Path) -> dict[str, str | None]:
	"""Minimal .env reader to avoid external runtime dependency."""
	values: dict[str, str | None] = {}
	for raw_line in path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#"):
			continue
		if line.startswith("export "):
			line = line[len("export ") :].strip()

		if "=" not in line:
			values[line] = None
			continue

		key, raw_value = line.split("=", 1)
		key = key.strip()
		value = raw_value.strip()
		if not key:
			continue

		if value and value[0] in {"\"", "'"} and value[-1:] == value[0]:
			try:
				value = ast.literal_eval(value)
			except Exception:
				value = value[1:-1]
		elif " #" in value:
			# Keep shell-style inline comments only when separated by whitespace.
			value = value.split(" #", 1)[0].rstrip()

		values[key] = "" if value is None else str(value)

	return values


def _is_sensitive(name: str) -> bool:
	upper = name.upper()
	return (
		upper.endswith("_KEY")
		or "API_KEY" in upper
		or upper.endswith("_SECRET")
		or upper.endswith("_PASSWORD")
		or upper.endswith("_PASS")
		or upper.endswith("_PWD")
		or upper.endswith("_TOKEN")
		or upper in {"DATABASE_URL", "GOOGLE_AI_API_KEY"}
	)


def _serialize(value: Any) -> str:
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, (list, dict, tuple, set)):
		return json.dumps(value, ensure_ascii=True)
	return str(value)


def _mask(key: str, value: str, show_secrets: bool) -> str:
	if show_secrets or not _is_sensitive(key):
		return value
	if not value:
		return value
	if len(value) <= 6:
		return "***"
	return value[:2] + "***" + value[-2:]


def _source_for(key: str, env_map: dict[str, str | None]) -> str:
	if key in os.environ:
		return "process_env"
	# Keep empty-string values as env_file to match pydantic-settings behavior.
	if key in env_map and env_map.get(key) is not None:
		return "env_file"
	return "default"


def _parse_settings_defaults_from_ast() -> dict[str, str]:
	"""Fallback parser when importing Settings fails (e.g. malformed env)."""
	result: dict[str, str] = {}
	config_file = SERVER_DIR / "app" / "core" / "config.py"
	if not config_file.exists():
		return result

	source = config_file.read_text(encoding="utf-8")
	tree = ast.parse(source)

	for node in tree.body:
		if isinstance(node, ast.ClassDef) and node.name == "Settings":
			for stmt in node.body:
				if not isinstance(stmt, ast.AnnAssign):
					continue
				if not isinstance(stmt.target, ast.Name):
					continue
				key = stmt.target.id
				if not key.isupper():
					continue

				if stmt.value is None:
					result[key] = "<required>"
					continue

				try:
					literal = ast.literal_eval(stmt.value)
					result[key] = _serialize(literal)
				except Exception:
					try:
						result[key] = ast.unparse(stmt.value)
					except Exception:
						result[key] = "<dynamic>"
			break

	return result


def _parse_settings_keys_from_ast() -> set[str]:
	"""Read declared uppercase Settings fields without importing app.core.config."""
	keys: set[str] = set()
	config_file = SERVER_DIR / "app" / "core" / "config.py"
	if not config_file.exists():
		return keys

	source = config_file.read_text(encoding="utf-8")
	tree = ast.parse(source)

	for node in tree.body:
		if isinstance(node, ast.ClassDef) and node.name == "Settings":
			for stmt in node.body:
				if not isinstance(stmt, ast.AnnAssign):
					continue
				if not isinstance(stmt.target, ast.Name):
					continue
				key = stmt.target.id
				if key.isupper():
					keys.add(key)
			break

	return keys


def _rows_typed(env_file: Path) -> tuple[list[dict[str, str]], str]:
	from app.core.config import Settings

	env_map = _dotenv_values(env_file)
	settings = Settings(_env_file=str(env_file))
	rows: list[dict[str, str]] = []

	for key in sorted(Settings.model_fields.keys()):
		field = Settings.model_fields[key]
		source = _source_for(key, env_map)
		effective = _serialize(getattr(settings, key))

		if field.is_required():
			default_str = "<required>"
		elif field.default_factory is not None:
			try:
				default_str = _serialize(field.default_factory())
			except Exception:
				default_str = "<default_factory>"
		else:
			default_str = _serialize(field.default)

		rows.append(
			{
				"key": key,
				"source": source,
				"effective": effective,
				"process_env_raw": "" if os.environ.get(key) is None else str(os.environ.get(key)),
				"env_file_raw": "" if env_map.get(key) is None else str(env_map.get(key)),
				"default": default_str,
			}
		)

	return rows, "typed"


def _rows_fallback(env_file: Path) -> tuple[list[dict[str, str]], str]:
	env_map = _dotenv_values(env_file)
	defaults = _parse_settings_defaults_from_ast()
	declared_keys = _parse_settings_keys_from_ast()

	if declared_keys:
		keys = sorted(set(defaults.keys()) | set(env_map.keys()) | {k for k in os.environ.keys() if k in declared_keys})
	else:
		keys = sorted(set(defaults.keys()) | set(env_map.keys()) | set(os.environ.keys()))
	keys = [k for k in keys if k.isupper()]

	rows: list[dict[str, str]] = []
	for key in keys:
		src = _source_for(key, env_map)
		process_raw = "" if os.environ.get(key) is None else str(os.environ.get(key))
		env_raw = "" if env_map.get(key) is None else str(env_map.get(key))
		default_val = defaults.get(key, "<unknown>")

		if src == "process_env":
			effective = process_raw
		elif src == "env_file":
			effective = env_raw
		else:
			effective = default_val

		rows.append(
			{
				"key": key,
				"source": src,
				"effective": effective,
				"process_env_raw": process_raw,
				"env_file_raw": env_raw,
				"default": default_val,
			}
		)

	return rows, "fallback"


def _load_rows(env_file: Path) -> tuple[list[dict[str, str]], str, str | None]:
	try:
		rows, mode = _rows_typed(env_file)
		return rows, mode, None
	except Exception as exc:
		rows, mode = _rows_fallback(env_file)
		return rows, mode, str(exc)


def _print_table(rows: list[dict[str, str]], env_file: Path, mode: str, show_secrets: bool) -> None:
	key_w = max(len("KEY"), *(len(r["key"]) for r in rows))
	src_w = max(len("SOURCE"), *(len(r["source"]) for r in rows))

	print(f"ENV FILE: {env_file}")
	print(f"RESOLUTION MODE: {mode}")
	print(f"TOTAL SETTINGS: {len(rows)}")
	print()
	print(f"{'KEY'.ljust(key_w)}  {'SOURCE'.ljust(src_w)}  EFFECTIVE")
	print(f"{'-' * key_w}  {'-' * src_w}  {'-' * 80}")

	for row in rows:
		value = _mask(row["key"], row["effective"], show_secrets)
		if len(value) > 160:
			value = value[:157] + "..."
		print(f"{row['key'].ljust(key_w)}  {row['source'].ljust(src_w)}  {value}")


def main() -> int:
	env_file = _default_env_file().expanduser().resolve()
	if not env_file.exists():
		print(f"[CHECK_SETTING] File not found: {env_file}")
		return 1

	rows, mode, typed_error = _load_rows(env_file)
	_print_table(rows, env_file, mode, show_secrets=False)
	if typed_error:
		print()
		print(f"NOTE: typed mode failed, fallback mode used: {typed_error}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
