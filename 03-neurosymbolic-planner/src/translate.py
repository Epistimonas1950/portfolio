"""Natural language -> problem instance, behind an interface, with the failure surface exposed.

This is the component the brief says is the whole failure surface, and the reason it is
written behind a `TranslationBackend` interface is that the interface is what makes the
failure *measurable*.  Everything downstream of `translate` is proved optimal.
Everything upstream of it is English.  Two implementations ship:

    RuleBackend     deterministic, offline, no model, no network.  A real parser for a
                    real (small) grammar: it works today and it is what the accuracy
                    number in the README is measured on.
    OllamaBackend   the language-model path.  Raises NotImplementedError naming exactly
                    what it needs, because there is no model on this machine and a mock
                    that returned plausible JSON would be a fabricated measurement.

THE GRAMMAR THE RULE BACKEND TARGETS, fixed before the corpus was scored:

    capacity     "the van holds four crates" | "capacity is 5" | "our bike can carry two"
    stops        any gazetteer name, with the nearest preceding quantity as its demand
    windows      "before/by/no later than T" -> latest       "after/from T" -> earliest
                 "between T and T" -> both;  attached to the nearest PRECEDING stop in
                 the same sentence, or globally when the phrase carries a scope marker
                 ("all", "everything", "both") or names no stop at all
    service      "each drop takes 5 minutes"
    start        "we start at 07:30"
    times        HH:MM | 9am | 3pm | noon | midday

Anything outside that grammar is a miss, and the point of the benchmark is to count the
misses and classify them, not to keep widening the grammar until the number looks good.

WHAT THE WARNINGS ARE FOR.  A parser that knows it is confused is worth far more than one
that is silently confident, so `RuleBackend` flags the constructions it can see but
cannot represent -- an unknown place name, a relative time ("before lunch"), a vague
quantity ("a couple"), a conditional, a negation, a coreference, an ordering constraint.
`bench/translation_accuracy.py` reports how many of the failures were self-flagged.  The
residue -- wrong, and confident about it -- is the number that should worry a deployer,
and it is reported separately for exactly that reason.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field

from .domain import Instance, instance_from_spec, load_gazetteer

DEFAULT_SERVICE = 6
DEFAULT_RELOAD = 10
DEFAULT_START = 8 * 60
DEFAULT_DAY_END = 18 * 60

NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
NUM = r"(?:\d+|" + "|".join(NUM_WORDS) + r")"
UNIT = r"(?:crates?|boxes?|parcels?|packages?|kgs?|kilograms?|kilos?)"
TIME = r"(?:\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm)|noon|midday)"

SCOPE_MARKER = re.compile(r"\b(?:all|everything|both|the lot)\b")

# Constructions the grammar cannot represent.  Detecting them is not parsing them; it
# is the parser admitting, in public, where its edge is.
SUSPICIOUS = [
    (re.compile(r"\b(?:lunch|first thing|this afternoon|this morning|tonight|"
                r"end of (?:the )?day|asap|later today|early|late)\b"), "relative_time"),
    (re.compile(r"\b(?:a couple|a few|some|several|a handful)\b"), "vague_quantity"),
    (re.compile(r"\b(?:same (?:window|as)|likewise|as above|ditto)\b"), "coreference"),
    (re.compile(r"\b(?:if|unless|otherwise|in case)\b"), "conditional"),
    (re.compile(r"\b(?:except|apart from|other than|but not)\b"), "exception_scope"),
    (re.compile(r"\b(?:nothing to|skip|don't|do not|no longer)\b"), "negation"),
    (re.compile(r"\b(?:first,|then|after that|in that order|before the)\b"),
     "unrepresentable_constraint"),
    (re.compile(r"\b(?:yesterday|last (?:week|time)|the usual|as before)\b"),
     "missing_context"),
    (re.compile(r"\b(?:half|third|quarter) (?:full|empty|loaded)\b"), "implicit_capacity"),
    (re.compile(r"\bsplit\b|\bshared? (?:between|among)\b|\bevenly\b"), "arithmetic"),
    (re.compile(r"\b(?:kgs?|kilograms?|kilos?)\b"), "unit_mismatch"),
]


class TranslationError(ValueError):
    """The request could not be turned into an instance.  Names what was missing."""


@dataclass
class TranslationResult:
    spec: dict | None
    warnings: list[str] = field(default_factory=list)
    backend: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.spec is not None


class TranslationBackend:
    """NL request -> instance spec.  One method; everything else is measurement."""

    name = "backend"

    def translate(self, text: str) -> TranslationResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def to_instance(self, text: str, name: str = "translated") -> Instance:
        res = self.translate(text)
        if not res.ok:
            raise TranslationError(res.error or "translation failed")
        return instance_from_spec(res.spec, self.gazetteer, name=name)


# --------------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------------- #

def _normalise(text: str) -> str:
    t = text.lower()
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t)


def _num(token: str) -> int:
    token = token.strip()
    return int(token) if token.isdigit() else NUM_WORDS[token]


def parse_clock(token: str) -> int:
    """'09:30' | '9am' | '3 pm' | 'noon' -> minutes from midnight."""
    token = token.strip()
    if token in ("noon", "midday"):
        return 12 * 60
    if ":" in token:
        h, m = token.split(":")
        return int(h) * 60 + int(m)
    m = re.match(r"(\d{1,2})\s*(am|pm)", token)
    if not m:
        raise TranslationError(f"unparsable time {token!r}")
    h = int(m.group(1)) % 12
    return (h + (12 if m.group(2) == "pm" else 0)) * 60


def _mask(text: str, span: tuple[int, int]) -> str:
    a, b = span
    return text[:a] + "#" * (b - a) + text[b:]


def canonical(spec: dict) -> dict:
    """Canonical form for exact-match scoring: ints, stops sorted by name."""
    return {
        "capacity": int(spec["capacity"]),
        "service_minutes": int(spec.get("service_minutes", DEFAULT_SERVICE)),
        "reload_minutes": int(spec.get("reload_minutes", DEFAULT_RELOAD)),
        "start_time": int(spec.get("start_time", DEFAULT_START)),
        "stops": sorted(({"name": s["name"], "demand": int(s["demand"]),
                          "earliest": int(s["earliest"]), "latest": int(s["latest"])}
                         for s in spec["stops"]), key=lambda s: s["name"]),
    }


def field_report(gold: dict, got: dict | None) -> dict:
    """Per-field agreement, so a near miss is not scored the same as nonsense."""
    if got is None:
        return {"capacity": 0, "stop_set": 0, "demands": 0, "windows": 0,
                "service": 0, "start": 0, "exact": 0}
    g, p = canonical(gold), canonical(got)
    gs = {s["name"]: s for s in g["stops"]}
    ps = {s["name"]: s for s in p["stops"]}
    same_set = set(gs) == set(ps)
    shared = set(gs) & set(ps)
    return {
        "capacity": int(g["capacity"] == p["capacity"]),
        "stop_set": int(same_set),
        "demands": int(same_set and all(gs[k]["demand"] == ps[k]["demand"] for k in shared)),
        "windows": int(same_set and all(gs[k]["earliest"] == ps[k]["earliest"]
                                        and gs[k]["latest"] == ps[k]["latest"]
                                        for k in shared)),
        "service": int(g["service_minutes"] == p["service_minutes"]),
        "start": int(g["start_time"] == p["start_time"]),
        "exact": int(g == p),
    }


# --------------------------------------------------------------------------------- #
# the offline backend
# --------------------------------------------------------------------------------- #

class RuleBackend(TranslationBackend):
    """A deterministic recursive-free parser for the grammar documented above.

    Pipeline, in order, because each stage masks the text it consumed so the next stage
    cannot re-read it -- "the van holds four crates" must not contribute a demand of
    four, and "before 11:00" must not contribute a quantity of 11:

        normalise -> start time -> service time -> capacity -> time windows
                  -> gazetteer names -> quantities -> attach
    """

    name = "rule"

    CAPACITY_PATTERNS = [
        re.compile(r"\b(?:van|bike|truck|vehicle|lorry)(?:'s)?\s+"
                   r"(?:can\s+carry|holds?|takes?|carries|capacity\s+is|capacity)\s+"
                   r"(" + NUM + r")"),
        re.compile(r"\bcapacity(?:\s+is)?(?:\s+of)?\s+(" + NUM + r")"),
        re.compile(r"\b(" + NUM + r")[-\s]crate\s+van\b"),
    ]
    SERVICE_PATTERN = re.compile(
        r"\beach\s+(?:drop|stop|delivery|call)\s+takes\s+(" + NUM + r")\s*min")
    START_PATTERN = re.compile(r"\b(?:we\s+)?start(?:s|ing)?\s+at\s+(" + TIME + r")")
    BETWEEN = re.compile(r"\bbetween\s+(" + TIME + r")\s+and\s+(" + TIME + r")")
    LATEST = re.compile(r"\b(?:before|by|no later than|not after|until)\s+(" + TIME + r")")
    EARLIEST = re.compile(r"\b(?:after|from|not before|no earlier than)\s+(" + TIME + r")")
    QUANTITY = re.compile(r"\b(" + NUM + r")\s*(?:" + UNIT + r")?\b")

    def __init__(self, gazetteer: dict[str, list[int]] | None = None,
                 day_end: int = DEFAULT_DAY_END) -> None:
        self.gazetteer = gazetteer if gazetteer is not None else load_gazetteer()
        self.day_end = day_end
        self.places = sorted((k for k in self.gazetteer if k != "depot"),
                             key=len, reverse=True)   # longest name first
        self._place_res = [(p, re.compile(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])"))
                           for p in self.places]

    # -- stage helpers ------------------------------------------------------------- #

    def _windows(self, w: str) -> tuple[str, list[dict]]:
        found: list[dict] = []
        for regex, kind in ((self.BETWEEN, "both"), (self.LATEST, "latest"),
                            (self.EARLIEST, "earliest")):
            while True:
                m = regex.search(w)
                if not m:
                    break
                entry = {"span": m.span(), "kind": kind}
                if kind == "both":
                    entry["earliest"] = parse_clock(m.group(1))
                    entry["latest"] = parse_clock(m.group(2))
                elif kind == "latest":
                    entry["latest"] = parse_clock(m.group(1))
                else:
                    entry["earliest"] = parse_clock(m.group(1))
                # Scope: a marker just before, or just after, promotes it to global.
                ctx = w[max(0, m.start() - 25):m.start()] + " " + w[m.end():m.end() + 15]
                entry["global"] = bool(SCOPE_MARKER.search(ctx))
                found.append(entry)
                w = _mask(w, m.span())
        found.sort(key=lambda e: e["span"][0])
        return w, found

    def _sentence_spans(self, w: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"[^.;]+", w) if m.group().strip()]

    # -- the parse ----------------------------------------------------------------- #

    def translate(self, text: str) -> TranslationResult:
        raw = _normalise(text)
        w = raw
        warnings = [label for regex, label in SUSPICIOUS if regex.search(raw)]

        start = DEFAULT_START
        m = self.START_PATTERN.search(w)
        if m:
            start = parse_clock(m.group(1))
            w = _mask(w, m.span())

        service = DEFAULT_SERVICE
        m = self.SERVICE_PATTERN.search(w)
        if m:
            service = _num(m.group(1))
            w = _mask(w, m.span())

        capacity = None
        for pattern in self.CAPACITY_PATTERNS:
            m = pattern.search(w)
            if m:
                capacity = _num(m.group(1))
                w = _mask(w, m.span())
                break
        if capacity is None:
            return TranslationResult(
                None, warnings, self.name,
                "no vehicle capacity found -- expected something like 'the van holds "
                "four crates' or 'capacity 5'")

        w, windows = self._windows(w)

        places: list[tuple[int, str]] = []
        for place, regex in self._place_res:
            for m in regex.finditer(w):
                places.append((m.start(), place))
                w = _mask(w, m.span())
        places.sort()
        if not places:
            return TranslationResult(
                None, warnings, self.name,
                "no known stop names found -- every stop must be in the gazetteer")

        quantities = [(m.start(), m.end(), _num(m.group(1)))
                      for m in self.QUANTITY.finditer(w)]

        mentions = []
        prev_pos = -1
        used = set()
        for pos, place in places:
            demand = None
            for i, (qs, qe, val) in enumerate(quantities):
                if i in used or qe > pos or qs <= prev_pos:
                    continue
                demand = val
                used.add(i)
            mentions.append({"name": place, "demand": demand,
                             "earliest": start, "latest": self.day_end})
            prev_pos = pos

        # Attach each window to the nearest preceding stop in the same sentence, unless
        # it is marked global or names no stop at all.
        sentences = self._sentence_spans(raw)
        for entry in windows:
            wa, _wb = entry["span"]
            sent = next((s for s in sentences if s[0] <= wa < s[1]), (0, len(raw)))
            in_sentence = [i for i, (pos, _p) in enumerate(places)
                           if sent[0] <= pos < sent[1]]
            before = [i for i in in_sentence if places[i][0] < wa]
            if entry["global"] or not before:
                targets = range(len(mentions))
            else:
                targets = [before[-1]]
            for i in targets:
                if "earliest" in entry:
                    mentions[i]["earliest"] = entry["earliest"]
                if "latest" in entry:
                    mentions[i]["latest"] = entry["latest"]

        # A stop named twice ("...two to Riverside Clinic. Riverside Clinic must be done
        # before 12:00.") is one stop.  Demands take the largest stated quantity rather
        # than the default that a bare re-mention would contribute, and the windows
        # intersect, which is the only reading under which both sentences are true.
        stops: dict[str, dict] = {}
        for m in mentions:
            cur = stops.get(m["name"])
            if cur is None:
                stops[m["name"]] = {**m, "demand": m["demand"] or 1}
                continue
            if m["demand"] is not None:
                cur["demand"] = max(cur["demand"], m["demand"])
            cur["earliest"] = max(cur["earliest"], m["earliest"])
            cur["latest"] = min(cur["latest"], m["latest"])

        spec = {"capacity": capacity, "service_minutes": service,
                "reload_minutes": DEFAULT_RELOAD, "start_time": start,
                "stops": sorted(stops.values(), key=lambda s: s["name"])}
        return TranslationResult(spec, warnings, self.name)


# --------------------------------------------------------------------------------- #
# the language-model backend
# --------------------------------------------------------------------------------- #

class OllamaBackend(TranslationBackend):
    """The open-weight-model path from BRIEF.md.  A stub, and it says so.

    The intended implementation is a constrained-decoding call: hand the model the JSON
    schema `domain.spec_from_instance` produces, force the sampler to stay inside that
    grammar, and validate the result with `instance_from_spec` before it goes anywhere
    near the search.  The architecture is what makes that safe -- a malformed or
    hallucinated instance is caught at the schema boundary, and a *well-formed but wrong*
    instance is exactly what the translation benchmark exists to count.

    It raises rather than falling back to `RuleBackend`, because a silent fallback would
    make the accuracy number in the README a measurement of something other than what it
    claims to measure.
    """

    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b-instruct",
                 host: str = "http://localhost:11434",
                 gazetteer: dict[str, list[int]] | None = None) -> None:
        self.model = model
        self.host = host
        self.gazetteer = gazetteer if gazetteer is not None else load_gazetteer()

    def translate(self, text: str) -> TranslationResult:
        raise NotImplementedError(
            "OllamaBackend needs a running Ollama server with structured/constrained "
            f"output (model {self.model!r} at {self.host}), and neither the server, the "
            "model weights, nor network access exist on this machine. "
            "The offline RuleBackend is the one the measurements in results/ use; see "
            "STATUS.md. To enable this path: install Ollama, `ollama pull "
            f"{self.model}`, and implement the JSON-schema-constrained call against "
            "domain.spec_from_instance's schema.")


BACKENDS = {"rule": RuleBackend, "ollama": OllamaBackend}


def load_corpus(path: str | pathlib.Path | None = None) -> dict:
    path = pathlib.Path(path) if path else (
        pathlib.Path(__file__).resolve().parents[1] / "data" / "nl_requests.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing -- it is the hand-written NL corpus "
                                "the translation benchmark scores against.")
    with open(path) as fh:
        return json.load(fh)
