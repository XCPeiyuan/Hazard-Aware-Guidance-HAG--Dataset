import re


CONTROLLED_SYNONYM_GROUPS = (
    frozenset({"bike", "bicycle"}),
    frozenset({"car", "vehicle"}),
)

CONTROLLED_SYNONYM_GROUPS_LOOSE = CONTROLLED_SYNONYM_GROUPS + (
    frozenset({"barricade", "barrier"}),
    frozenset({"fence", "barrier"}),
    frozenset({"hole", "pit"}),
    frozenset({"manhole", "pit"}),
    frozenset({"opening", "pit"}),
    frozenset({"pole", "post"}),
    frozenset({"signpost", "pole"}),
    frozenset({"beam", "bar"}),
    frozenset({"crossbar", "beam"}),
    frozenset({"truck", "vehicle"}),
    frozenset({"van", "vehicle"}),
    frozenset({"motorcycle", "bike"}),
    frozenset({"obstacle", "barrier"}),
    frozenset({"debris", "brick"}),
    frozenset({"rubble", "brick"}),
    frozenset({"sign", "board"}),
    frozenset({"advertisement", "sign"}),
    frozenset({"box", "container"}),
    frozenset({"bin", "container"}),
    frozenset({"trash can", "container"}),
    frozenset({"cart", "trolley"}),
    frozenset({"pallet truck", "cart"}),
    frozenset({"step", "stair"}),
    frozenset({"wire", "cable"}),
    frozenset({"rope", "cable"}),
    frozenset({"hydrant", "pipe"}),
    frozenset({"antenna", "pole"}),
    frozenset({"scaffolding", "barrier"}),
)

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_SAFE_IRREGULAR_PLURALS = {"buses": "bus"}
_INVARIANT_OR_FALSE_COLLISION_NOUNS = frozenset(
    {"species", "news", "status", "glass", "glasses", "gas", "bus"}
)


def _singularize_token(token: str) -> str:
    if token in _SAFE_IRREGULAR_PLURALS:
        return _SAFE_IRREGULAR_PLURALS[token]
    if token in _INVARIANT_OR_FALSE_COLLISION_NOUNS:
        return token
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes", "sses")):
        return token[:-2]
    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is", "ses"))
    ):
        return token[:-1]
    return token


def normalize_name(name: str) -> str:
    """Return a cautious, auditable lexical normalization of an object name."""
    words = _NON_ALPHANUMERIC.sub(" ", name.lower()).split()
    return " ".join(_singularize_token(word) for word in words)


def are_controlled_synonyms(left: str, right: str, *, profile: str = "strict") -> bool:
    """Return whether two already-normalized names are an approved synonym pair."""
    if left == right:
        return False
    groups = (
        CONTROLLED_SYNONYM_GROUPS_LOOSE if profile == "loose"
        else CONTROLLED_SYNONYM_GROUPS
    )
    return any({left, right} == group for group in groups)
