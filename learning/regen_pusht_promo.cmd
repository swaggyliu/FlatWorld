@echo off
REM Regen Push-T promo mp4 once a stable torch+learning env is available.
REM On this Windows host, WMI/CIM and long PowerShell torch probes often exit
REM 4294967295; prefer this short cmd launcher with CUDA off and unbuffered IO.
REM
REM Existing shipped clips (keep until regen succeeds):
REM   docs/media/pusht_promo.mp4
REM   docs/media/promo_supermimic.mp4
REM   docs/media/dance_overlay.mp4
REM
REM Usage (from repo root, after activating a venv with flatworld[learning]):
REM   learning\regen_pusht_promo.cmd
REM   learning\regen_pusht_promo.cmd path\to\python.exe

setlocal
set "PY=%~1"
if "%PY%"=="" set "PY=python"
set CUDA_VISIBLE_DEVICES=
set PYTHONUNBUFFERED=1

echo Using: %PY%
"%PY%" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || (
  echo ERROR: torch import failed. Activate a learning venv first.
  exit /b 1
)

"%PY%" -u -m learning.render_pusht_promo ^
  --policy learning/checkpoints/push_t_wm_rsl/best_tol012_90pct.pt ^
  --wm learning/checkpoints/push_t ^
  --tol 0.12 --need-ok 5 --max-tries 40 ^
  --out-dir docs/media --out docs/media/pusht_promo.mp4

exit /b %ERRORLEVEL%
