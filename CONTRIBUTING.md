# Contributing to Flat World

Thank you for your interest in contributing!

## Development setup

```bash
git clone <your-fork-url>
cd miniEngine
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Running tests

```bash
# Full 2D test suite (headless)
HEADLESS=1 pytest test2D -q

# Single test
pytest test2D/test_2Dfem_elastic.py -v
```

CI runs on Ubuntu with `HEADLESS=1` and `CI=true` (no display required).

## Code style

- Match the surrounding module: naming, imports, and Taichi kernel patterns.
- Keep changes focused; avoid unrelated refactors in the same PR.
- Add or update tests when fixing bugs or adding features.

## Pull requests

1. Fork the repository and create a feature branch.
2. Ensure `pytest test2D` passes locally.
3. Describe what changed and why in the PR body.
4. Link related issues if any.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
