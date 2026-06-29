# Auto-loaded by Python when this directory is first on PYTHONPATH.
# Installs the lerobot.common -> lerobot redirect so RLinf's RECAP stages
# (compute_advantages.py, value training) import cleanly under lerobot>=0.3.
try:
    from pt4fm.compat import (
        install_lerobot_common_shim,
        install_lerobot_pyav_video_fallback,
    )

    install_lerobot_common_shim()
    install_lerobot_pyav_video_fallback()
except Exception as _e:  # never break the interpreter if pt4fm isn't importable
    pass
