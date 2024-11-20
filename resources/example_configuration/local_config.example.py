"""You can customize an equivalent file in order to control PyARPES behavior."""

SETTINGS = {
    "interactive": {
        "main_width": 600,
        "marginal_width": 300,
        "palette": "magma",
    },
    "use_tex": True,
}

CONFIG = {
    "WORKSPACE": {},
    "CURRENT_CONTEXT": None,
    "ENABLE_LOGGING": True,
    "LOGGING_STARTED": False,
    "LOGGING_FILE": None,
}

PLUGINS: set[str] = set()
