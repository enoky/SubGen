# Python / uv

This project uses uv.

Use:

- `uv sync` to install dependencies
- `uv run pytest` to run tests
- `uv run python -m <module>` to run Python modules
- `uv add <package>` to add dependencies

Do not use bare `python`, `pip`, or `.venv\Scripts\activate` unless explicitly requested.
Do not manually edit dependency pins; use `uv add`, `uv remove`, and `uv lock`.