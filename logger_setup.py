"""
logger_setup.py — legacy entry point kept for backward compatibility.

Old code called initialize_logger() directly (e.g. Pronunciation/run_example.py).
Now delegates to api.logging_config.setup_logging() so all rotation
configuration is in one place.  Falls back to a basic console handler
if the API stack is not importable (standalone scripts / tests).
"""
import logging
import os


def initialize_logger(log_dir: str = "logs") -> None:
    """Configure root logger with rotating file + console output.

    Delegates to api.logging_config.setup_logging() when available.
    Falls back to a simple RotatingFileHandler so standalone scripts
    (e.g. Pronunciation/run_example.py) still work without the full
    API stack installed.
    """
    try:
        from api.config import APP_ENV, LOG_DIR  # noqa: PLC0415
        from api.logging_config import setup_logging  # noqa: PLC0415
        setup_logging(log_dir=LOG_DIR, app_env=APP_ENV)
        return
    except ImportError:
        pass

    # Minimal fallback for standalone scripts
    from logging.handlers import RotatingFileHandler

    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        filename    = os.path.join(log_dir, "app.log"),
        maxBytes    = 20 * 1024 * 1024,  # 20 MB (consistent with api.logging_config)
        backupCount = 30,
        encoding    = "utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)