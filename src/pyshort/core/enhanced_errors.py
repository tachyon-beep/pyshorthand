"""Enhanced error messages with suggestions.

Provides helpful, actionable error messages for common mistakes.
"""


from pyshort.core.symbols import (
    VALID_LAYERS,
    VALID_LOCATIONS,
    VALID_RISK_LEVELS,
    VALID_ROLES,
    VALID_TAG_BASES,
    VALID_TYPES,
)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_close_matches(
    word: str, possibilities: list[str], n: int = 3, cutoff: float = 0.6
) -> list[str]:
    """Find close matches using Levenshtein distance.

    Args:
        word: Word to find matches for
        possibilities: List of possible matches
        n: Maximum number of matches to return
        cutoff: Minimum similarity ratio (0.0 to 1.0)

    Returns:
        List of close matches
    """
    if not word or not possibilities:
        return []

    # Calculate distances
    distances = []
    for possibility in possibilities:
        distance = levenshtein_distance(word.lower(), possibility.lower())
        max_len = max(len(word), len(possibility))
        similarity = 1.0 - (distance / max_len)
        if similarity >= cutoff:
            distances.append((similarity, possibility))

    # Sort by similarity and return top n
    distances.sort(reverse=True)
    return [match for _, match in distances[:n]]


def suggest_did_you_mean(word: str, category: str) -> str | None:
    """Generate "Did you mean?" suggestion.

    Args:
        word: Invalid word
        category: Category of word (role, layer, risk, tag, type, location)

    Returns:
        Formatted suggestion or None
    """
    possibilities_map = {
        "role": list(VALID_ROLES),
        "layer": list(VALID_LAYERS),
        "risk": list(VALID_RISK_LEVELS),
        "tag": list(VALID_TAG_BASES),
        "type": list(VALID_TYPES),
        "location": list(VALID_LOCATIONS),
    }

    possibilities = possibilities_map.get(category, [])
    if not possibilities:
        return None

    matches = find_close_matches(word, possibilities, n=3)
    if not matches:
        return None

    if len(matches) == 1:
        return f"Did you mean '{matches[0]}'?"
    else:
        options = "', '".join(matches)
        return f"Did you mean one of: '{options}'?"
