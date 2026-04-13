from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
	sys.path.insert(0, str(SERVER_DIR))


def _default_env_file() -> Path:
	project_env = SERVER_DIR.parent / ".env"
	if project_env.exists():
		return project_env
	return SERVER_DIR / ".env"


def _is_sensitive(name: str) -> bool:
	upper = name.upper()
	return (
		upper.endswith("_KEY")
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
	if key in env_map and env_map.get(key) not in (None, ""):
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


def _rows_typed(env_file: Path) -> tuple[list[dict[str, str]], str]:
	from app.core.config import Settings

	env_map = dotenv_values(env_file)
	settings = Settings(_env_file=str(env_file))
	rows: list[dict[str, str]] = []

	for key in sorted(Settings.model_fields.keys()):
		field = Settings.model_fields[key]
		source = _source_for(key, env_map)
		effective = _serialize(getattr(settings, key))

		default_raw = field.default
		if default_raw is None:
			default_str = "None"
		else:
			default_str = _serialize(default_raw)

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
	env_map = dotenv_values(env_file)
	defaults = _parse_settings_defaults_from_ast()

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


def _print_one(rows: list[dict[str, str]], key: str, env_file: Path, mode: str, show_secrets: bool) -> int:
	target = key.strip()
	if not target:
		print("[CHECK_SETTING] --key cannot be empty")
		return 1

	row = next((r for r in rows if r["key"] == target), None)
	if row is None:
		print(f"[CHECK_SETTING] Unknown key: {target}")
		print("[CHECK_SETTING] Available keys:")
		for r in rows:
			print(f"  - {r['key']}")
		return 1

	print(f"ENV FILE: {env_file}")
	print(f"RESOLUTION MODE: {mode}")
	print(f"KEY: {target}")
	print(f"SOURCE: {row['source']}")
	print(f"EFFECTIVE: {_mask(target, row['effective'], show_secrets)}")
	process = row["process_env_raw"] if row["process_env_raw"] else "(not set)"
	env_raw = row["env_file_raw"] if row["env_file_raw"] else "(not set)"
	print(f"PROCESS_ENV_RAW: {_mask(target, process, show_secrets)}")
	print(f"ENV_FILE_RAW: {_mask(target, env_raw, show_secrets)}")
	print(f"DEFAULT: {_mask(target, row['default'], show_secrets)}")
	return 0


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Inspect runtime settings and show value source (process_env/env_file/default)"
	)
	parser.add_argument(
		"--env-file",
		default=str(_default_env_file()),
		help="Path to env file (default: server/.env or project .env if present)",
	)
	parser.add_argument(
		"--key",
		help="Inspect one setting key only (example: NEXUSRAG_ENABLE_KG)",
	)
	parser.add_argument(
		"--format",
		choices=["table", "json"],
		default="table",
		help="Output format",
	)
	parser.add_argument(
		"--show-secrets",
		action="store_true",
		help="Show full values for sensitive keys",
	)
	args = parser.parse_args()

	env_file = Path(args.env_file).expanduser().resolve()
	if not env_file.exists():
		print(f"[CHECK_SETTING] File not found: {env_file}")
		return 1

	rows, mode, typed_error = _load_rows(env_file)

	if args.key:
		return _print_one(rows, args.key, env_file, mode, args.show_secrets)

	if args.format == "json":
		payload: dict[str, Any] = {
			"env_file": str(env_file),
			"resolution_mode": mode,
			"total": len(rows),
			"settings": [
				{
					"key": r["key"],
					"source": r["source"],
					"effective": _mask(r["key"], r["effective"], args.show_secrets),
					"process_env_raw": _mask(r["key"], r["process_env_raw"], args.show_secrets),
					"env_file_raw": _mask(r["key"], r["env_file_raw"], args.show_secrets),
					"default": _mask(r["key"], r["default"], args.show_secrets),
				}
				for r in rows
			],
		}
		if typed_error:
			payload["typed_resolution_error"] = typed_error
		print(json.dumps(payload, ensure_ascii=True, indent=2))
		return 0

	_print_table(rows, env_file, mode, args.show_secrets)
	if typed_error:
		print()
		print(f"NOTE: typed mode failed, fallback mode used: {typed_error}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
