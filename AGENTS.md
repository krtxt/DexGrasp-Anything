# Repository Guidelines

## Project Structure & Module Organization

- `train.py`, `train_ddm.py`: training entrypoints (single/multi‑GPU via scripts).
- `scripts/grasp_gen_ur/`: run wrappers for train/sample/test (recommended way to launch jobs).
- `configs/`: experiment configuration (model, optimizer, diffuser, task). Keep new configs additive and named by purpose (e.g., `configs/model/shadowhand_large.yaml`).
- `models/`, `utils/`, `datasets/`, `envs/`: core Python code (models, helpers, dataloaders, Isaac Gym tasks/assets).
- `assets/` and `envs/assets/`: URDF/meshes and simulation assets.
- `Process_your_dataset/`: utilities to build custom datasets (PCD sampling, URDF generation).
- `ckpts/` and `outputs/`: checkpoints and generated results (do not commit large artifacts).

## Build, Test, and Development Commands

- Create environment + install deps (see `README.md`): `conda create -n DGA python=3.8` then `pip install -r requirements.txt`.
- Build local third‑party deps under `src/` (CSDF, `pytorch3d`, Isaac Gym) as documented in `README.md`.
- Train: `bash scripts/grasp_gen_ur/train.sh <EXP_NAME>` (single GPU) or `bash scripts/grasp_gen_ur/train_ddm.sh <EXP_NAME>` (multi‑GPU).
- Sample: `bash scripts/grasp_gen_ur/sample.sh outputs/<exp_dir> [OPT]` (`[OPT]` enables physics‑guided sampling).
- Evaluate: `bash scripts/grasp_gen_ur/test.sh outputs/<exp_dir>/eval/<...>` (configure dataset paths in `envs/tasks/grasp_test_force_shadowhand.py` and `scripts/grasp_gen_ur/test.py`).

## Coding Style & Naming Conventions

- Python: 4‑space indentation, `snake_case` for functions/vars, `PascalCase` for classes, constants in `UPPER_SNAKE_CASE`.
- Prefer small, reviewable diffs: avoid drive‑by reformatting; keep changes scoped to the feature/bug.
- Keep paths configurable: do not hardcode machine‑local absolute paths in code or configs.

## Testing Guidelines

- This repo primarily uses script-based evaluation rather than a unit-test suite.
- When changing training/sampling/eval logic, rerun the smallest relevant command (e.g., `sample.sh` on a tiny config) and include the exact command and key logs/metrics in your PR.

## Commit & Pull Request Guidelines

- Commit messages in history are short and imperative (e.g., `init`, `new dataset`, “better visualization…”). Use the same style but be specific: `fix eval path handling`, `add realdex config`.
- PRs should include: purpose, how to reproduce (commands + config), expected GPU/Isaac Gym requirements, and screenshots/plots for visualization changes.

## Notes for Automation/Agents

- Be careful with symlinks (e.g., `data/`, `src/`) in local checkouts; prefer repo-relative paths and document any required external directories.
