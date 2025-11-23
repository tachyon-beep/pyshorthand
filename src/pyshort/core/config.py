"""Configuration file loading for PyShorthand.

Supports .pyshortrc files in INI format (zero dependencies).
Searches for config in: current dir → parent dirs → home dir
"""

from configparser import ConfigParser
from pathlib import Path
from typing import Any


def find_config_file(start_path: Path | None = None) -> Path | None:
    """
    Find .pyshortrc config file by walking up directory tree.

    Search order:
    1. Current directory
    2. Parent directories (walk up to root)
    3. Home directory (~/.pyshortrc)

    Args:
        start_path: Directory to start search from (default: cwd)

    Returns:
        Path to config file, or None if not found
    """
    if start_path is None:
        start_path = Path.cwd()

    # Walk up directory tree
    current = start_path.resolve()
    while True:
        config_path = current / ".pyshortrc"
        if config_path.exists():
            return config_path

        # Check if we've reached the root
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Check home directory
    home_config = Path.home() / ".pyshortrc"
    if home_config.exists():
        return home_config

    return None


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load configuration from .pyshortrc file.

    Args:
        config_path: Explicit path to config file (default: search)

    Returns:
        Dictionary of configuration options

    Example .pyshortrc:
        [format]
        indent = 2
        align_types = true
        prefer_unicode = true
        sort_state_by = location
        max_line_length = 100

        [lint]
        strict = false
        max_line_length = 120
    """
    if config_path is None:
        config_path = find_config_file()

    if config_path is None:
        return {}

    parser = ConfigParser()
    parser.read(config_path)

    config = {}

    # Parse [format] section
    if parser.has_section("format"):
        config["format"] = {
            "indent": parser.getint("format", "indent", fallback=2),
            "align_types": parser.getboolean("format", "align_types", fallback=True),
            "prefer_unicode": parser.getboolean("format", "prefer_unicode", fallback=True),
            "sort_state_by": parser.get("format", "sort_state_by", fallback="location"),
            "max_line_length": parser.getint("format", "max_line_length", fallback=100),
        }

    # Parse [lint] section
    if parser.has_section("lint"):
        config["lint"] = {
            "strict": parser.getboolean("lint", "strict", fallback=False),
            "max_line_length": parser.getint("lint", "max_line_length", fallback=120),
        }

    # Parse [viz] section
    if parser.has_section("viz"):
        config["viz"] = {
            "direction": parser.get("viz", "direction", fallback="TB"),
            "show_state_vars": parser.getboolean("viz", "show_state_vars", fallback=True),
            "show_methods": parser.getboolean("viz", "show_methods", fallback=True),
            "show_dependencies": parser.getboolean("viz", "show_dependencies", fallback=True),
            "color_by_risk": parser.getboolean("viz", "color_by_risk", fallback=True),
        }

    return config


def merge_config_with_args(config: dict[str, Any], args: Any, section: str) -> dict[str, Any]:
    """
    Merge config file settings with CLI arguments.

    CLI arguments take precedence over config file.

    Args:
        config: Loaded configuration dict
        args: Parsed argparse Namespace
        section: Config section to merge ("format", "lint", etc.)

    Returns:
        Merged configuration dict
    """
    section_config = config.get(section, {})
    merged = section_config.copy()

    # Override with CLI args (only if explicitly provided)
    args_dict = vars(args)
    for key, value in args_dict.items():
        # Skip None values (not explicitly set)
        if value is not None:
            merged[key] = value

    return merged


def create_default_config(path: Path) -> None:
    """
    Create a default .pyshortrc config file.

    Args:
        path: Path where config should be created
    """
    default_config = """# PyShorthand Configuration File
# See: https://github.com/tachyon-beep/animated-system

[format]
# Indentation (spaces)
indent = 2

# Align type annotations vertically
align_types = true

# Use Unicode symbols (∈, →) instead of ASCII (IN, ->)
prefer_unicode = true

# Sort state variables by: location, name, or none
sort_state_by = location

# Maximum line length
max_line_length = 100

[lint]
# Treat warnings as errors
strict = false

# Maximum line length for linting
max_line_length = 120

[viz]
# Default diagram direction: TB, LR, RL, BT
direction = TB

# Show state variables in diagrams
show_state_vars = true

# Show methods in class diagrams
show_methods = true

# Show dependency edges
show_dependencies = true

# Color nodes by risk level
color_by_risk = true
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(default_config)


if __name__ == "__main__":
    # Test config loading
    config_path = find_config_file()
    if config_path:
        print(f"Found config: {config_path}")
        config = load_config(config_path)
        print("Loaded config:")
        for section, values in config.items():
            print(f"  [{section}]")
            for key, value in values.items():
                print(f"    {key} = {value}")
    else:
        print("No config file found")
