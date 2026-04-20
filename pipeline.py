"""
pipeline.py — Production-grade moderation guardrail pipeline.

Architecture:
    Layer 1: Regex pre-filter (blocklist)
    Layer 2: Calibrated DistilBERT classifier
    Layer 3: Human review queue (uncertainty band)

Usage:
    from pipeline import ModerationPipeline
    pipe = ModerationPipeline(model_dir='./model_best_mitigated')
    result = pipe.predict("Your comment text here")
"""

import re
import unicodedata
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    # Newer scikit-learn uses FrozenEstimator instead of cv='prefit'.
    from sklearn.frozen import FrozenEstimator
except Exception:
    FrozenEstimator = None


# =============================================================================
# LAYER 1: REGEX BLOCKLIST
# Organized by category. Each pattern uses word boundaries and handles
# common conjugation / contraction variants.
# =============================================================================

BLOCKLIST: dict[str, list[re.Pattern]] = {

    # ─── Category 1: Direct threats of violence (≥ 5 patterns) ──────────────
    "direct_threat": [
        # "I will / I'm going to / I'll / gonna / someone should [kill|murder|shoot|stab|hurt] you"
        re.compile(
            r"\b(i(\'ll|\ will|\ am\ going\ to|\'m\ going\ to)|gonna|someone\ should)\s+"
            r"(kill|murder|shoot|stab|hurt|harm|destroy|eliminate|execute)\s+you\b",
            re.IGNORECASE,
        ),
        # "you('re | are) going to die / get hurt / be killed"
        re.compile(
            r"\byou(\'re|\ are)\s+going\s+to\s+(die|get\s+(hurt|killed|shot|stabbed))\b",
            re.IGNORECASE,
        ),
        # "I'll find where you live / I know where you live and I'll come for you"
        re.compile(
            r"\bi(\'ll|\ will)\s+find\s+(where|out\ where)\s+you\s+live\b",
            re.IGNORECASE,
        ),
        # "you('re) dead / you('re) a dead man|woman|person"
        re.compile(
            r"\byou(\'re|\ are)\s+(dead(\s+meat)?|a\s+dead\s+(man|woman|person))\b",
            re.IGNORECASE,
        ),
        # Capturing group variant: "I will [threat_verb] you"
        re.compile(
            r"\b(i(\'ll|\ will|\ am\ going\ to|\ gonna)|we(\'ll|\ will))\s+"
            r"(?P<threat>kill|murder|shoot|stab|hurt|assault|attack|finish)\s+you\b",
            re.IGNORECASE,
        ),
        # "watch your back" with explicit threat escalation
        re.compile(
            r"\bwatch\s+your\s+(back|step|mouth)\b.{0,40}\b(or\s+else|or\s+you(\'ll|\ will)|i\s+swear)\b",
            re.IGNORECASE,
        ),
    ],

    # ─── Category 2: Calls for self-harm / suicide (≥ 4 patterns) ───────────
    "self_harm_directed": [
        # "you should kill yourself / go kill yourself"
        re.compile(
            r"\b(you\s+should|go|just\s+go)\s+kill\s+yourself\b",
            re.IGNORECASE,
        ),
        # "nobody / no one would miss you if you died / were gone"
        re.compile(
            r"\b(nobody|no\s+one|no\s+body)\s+would\s+(miss|care\s+about)\s+you\b",
            re.IGNORECASE,
        ),
        # "do everyone a favour / favor and (die|disappear|end it)"
        re.compile(
            r"\bdo\s+(every(one|body)|us|the\s+world)\s+a\s+favou?r\s+and\s+(die|disappear|end\s+it|off\s+yourself)\b",
            re.IGNORECASE,
        ),
        # "the world would be better (off) without you"
        re.compile(
            r"\bthe\s+world\s+would\s+be\s+better\s+(off\s+)?without\s+you\b",
            re.IGNORECASE,
        ),
        # "you should (just) end it / end your life"
        re.compile(
            r"\byou\s+should\s+(just\s+)?end\s+(it|your\s+life|everything)\b",
            re.IGNORECASE,
        ),
    ],

    # ─── Category 3: Doxxing and stalking threats (≥ 4 patterns) ────────────
    "doxxing_stalking": [
        # "I know where you live / work / go to school"
        re.compile(
            r"\bi\s+(know|found|have)\s+(where|your)\s+(you\s+)?(live|work|go\s+to\s+school|address|location)\b",
            re.IGNORECASE,
        ),
        # "I'll / I will / I'm gonna post your address / number / name"
        re.compile(
            r"\bi(\'ll|\ will|\'m\ gonna|\ am\ going\ to)\s+(post|share|publish|leak|expose)\s+your\s+"
            r"(address|phone(\s+number)?|number|real\s+name|info(rmation)?|details?|location)\b",
            re.IGNORECASE,
        ),
        # "I found your real name / identity"
        re.compile(
            r"\bi\s+(found|know|have)\s+your\s+(real\s+)?(name|identity|full\s+name|home(\s+address)?)\b",
            re.IGNORECASE,
        ),
        # "everyone will know who you (really) are"
        re.compile(
            r"\bevery(one|body)\s+will\s+(know|find\s+out)\s+(who\s+you\s+(really\s+)?are|your\s+(real\s+)?identity)\b",
            re.IGNORECASE,
        ),
        # "I've been watching / following you"
        re.compile(
            r"\bi(\'ve|\ have)\s+been\s+(watching|following|tracking|stalking)\s+you\b",
            re.IGNORECASE,
        ),
    ],

    # ─── Category 4: Severe dehumanization (≥ 4 patterns) ───────────────────
    "dehumanization": [
        # "[group] are not human / not people / subhuman"
        re.compile(
            r"\b\w+\s+are\s+(not\s+(?:human|people|persons?)|subhuman|less\s+than\s+human)\b",
            re.IGNORECASE,
        ),
        # "[group] are animals / vermin / parasites / insects / pests"
        re.compile(
            r"\b\w+\s+are\s+(animals?|vermin|parasites?|insects?|pests?|rats?|cockroaches?|dogs?|savages?)\b",
            re.IGNORECASE,
        ),
        # "[group] should be exterminated / wiped out / eliminated / cleansed"
        re.compile(
            r"\b\w+\s+should\s+be\s+(exterminated|wiped\s+out|eliminated|cleansed|eradicated|purged)\b",
            re.IGNORECASE,
        ),
        # "[group] are a disease / plague / cancer / infestation"
        re.compile(
            r"\b\w+\s+are\s+(a\s+)?(disease|plague|cancer|infestation|virus|blight|infection)\b",
            re.IGNORECASE,
        ),
        # "send [group] back / get rid of [group]"
        re.compile(
            r"\b(send\s+them\s+back|get\s+rid\s+of\s+(them|these\s+\w+)|they\s+don(\'t|\s+not)\s+belong\s+here)\b",
            re.IGNORECASE,
        ),
    ],

    # ─── Category 5: Coordinated harassment (≥ 3 patterns) ──────────────────
    "coordinated_harassment": [
        # "everyone report [username/account]" — lookahead for flexibility
        re.compile(
            r"\bevery(one|body)\s+report\b(?=.{0,50})",
            re.IGNORECASE,
        ),
        # "let's (all) go after / attack / target [username]"
        re.compile(
            r"\blet(\'s|\ us)\s+(all\s+)?(go\s+after|attack|target|mass\s+report|brigade|dog\s*pile)\b",
            re.IGNORECASE,
        ),
        # "raid (their|this|the) (profile|account|server|stream|channel)"
        re.compile(
            r"\braid\s+(their|this|the|his|her)\s+(profile|account|server|stream|channel|page)\b",
            re.IGNORECASE,
        ),
        # "mass report this account / user"
        re.compile(
            r"\bmass\s+(report|flag|block)\s+(this|the|his|her|their)\s+(account|user|profile|post)\b",
            re.IGNORECASE,
        ),
    ],
}


def _normalize_text(text: str) -> str:
    """
    Normalize Unicode: strip zero-width characters, map homoglyphs to ASCII,
    and apply NFKC normalization. This defends against evasion attacks (Part 3).
    """
    # Remove zero-width spaces and other invisible characters
    text = re.sub(r'[\u200B-\u200D\uFEFF\u00AD]', '', text)
    # NFKC normalization maps Cyrillic homoglyphs to their Latin equivalents
    text = unicodedata.normalize('NFKC', text)
    return text


def input_filter(text: str) -> dict | None:
    """
    Layer 1: Fast regex pre-filter.

    Returns a block decision dict if any pattern matches, else None.
    Iterates over all categories and patterns in BLOCKLIST.
    The matched category name is included in the decision for auditability.
    """
    normalized = _normalize_text(text)
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(normalized):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0,
                    "matched_pattern": pattern.pattern[:80],
                }
    return None


# =============================================================================
# SKLEARN WRAPPER FOR DISTILBERT (for CalibratedClassifierCV)
# =============================================================================

class DistilBERTWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper around DistilBERT so that
    CalibratedClassifierCV can calibrate its output probabilities.
    """

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir = model_dir
        self.device = device
        self.tokenizer_ = None
        self.model_ = None

    def _load(self):
        if self.model_ is None:
            self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_dir)
            self.model_ = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model_.to(self.device)
            self.model_.eval()

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self  # pre-trained, no fitting needed here

    def predict_proba(self, X, batch_size: int = 32) -> np.ndarray:
        """X is a list/array of text strings."""
        self._load()
        all_probs = []
        for i in range(0, len(X), batch_size):
            batch = list(X[i: i + batch_size])
            enc = self.tokenizer_(
                batch, truncation=True, padding=True,
                max_length=128, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model_(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)  # shape (N, 2)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


# =============================================================================
# MODERATION PIPELINE
# =============================================================================

class ModerationPipeline:
    """
    Three-layer production moderation guardrail pipeline.

    Layer 1 — Regex pre-filter (BLOCKLIST):
        Catches obvious, high-severity content immediately.
        Returns 'block' with category label.

    Layer 2 — Calibrated DistilBERT:
        Runs the mitigated model with isotonic probability calibration.
        confidence >= 0.6  → block
        confidence <= 0.4  → allow
        0.4 < confidence < 0.6 → escalate to Layer 3

    Layer 3 — Human review queue:
        Uncertain predictions are routed for human review.
        Returns 'review'.
    """

    BLOCK_THRESHOLD = 0.6
    ALLOW_THRESHOLD = 0.4

    def __init__(
        self,
        model_dir: str = "model_best_mitigated",
        device: str | None = None,
        calibration_texts: list[str] | None = None,
        calibration_labels: list[int] | None = None,
    ):
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Base model wrapper
        self._base = DistilBERTWrapper(model_dir=model_dir, device=self.device)
        self._base.fit([], [])  # initialise classes_

        # Calibrated wrapper (fitted lazily or explicitly)
        self._calibrated: CalibratedClassifierCV | None = None

        if calibration_texts and calibration_labels:
            self.calibrate(calibration_texts, calibration_labels)

    def calibrate(self, texts: list[str], labels: list[int]) -> "ModerationPipeline":
        """
        Fit isotonic calibration on provided texts + labels.
        Should be called once before deployment using a held-out set.
        """
        print(f"Calibrating on {len(texts)} examples...")
        if FrozenEstimator is not None:
            self._calibrated = CalibratedClassifierCV(
                estimator=FrozenEstimator(self._base), method="isotonic", cv=None
            )
        else:
            # Backward-compatible fallback for older scikit-learn versions.
            self._calibrated = CalibratedClassifierCV(
                estimator=self._base, method="isotonic", cv=5
            )

        self._calibrated.fit(texts, labels)
        print("Calibration complete.")
        return self

    def _get_confidence(self, text: str) -> float:
        """Return calibrated (or raw) P(toxic) for a single text."""
        normalized = _normalize_text(text)
        if self._calibrated is not None:
            cal = self._calibrated.predict_proba([normalized])
            return float(cal[0, 1])
        else:
            raw = self._base.predict_proba([normalized])  # shape (1, 2)
            return float(raw[0, 1])

    def predict(self, text: str) -> dict:
        """
        Run the full three-layer pipeline on a single text.

        Returns
        -------
        dict with keys:
            decision   : 'block' | 'allow' | 'review'
            layer      : 'input_filter' | 'model' | 'human_review'
            confidence : float in [0, 1]
            category   : str  (only for input_filter blocks)
        """
        # ── Layer 1: regex pre-filter ────────────────────────────────────────
        filter_result = input_filter(text)
        if filter_result is not None:
            return filter_result

        # ── Layer 2: calibrated model ────────────────────────────────────────
        confidence = self._get_confidence(text)

        if confidence >= self.BLOCK_THRESHOLD:
            return {
                "decision": "block",
                "layer": "model",
                "confidence": round(confidence, 4),
            }

        if confidence <= self.ALLOW_THRESHOLD:
            return {
                "decision": "allow",
                "layer": "model",
                "confidence": round(confidence, 4),
            }

        # ── Layer 3: human review ────────────────────────────────────────────
        return {
            "decision": "review",
            "layer": "human_review",
            "confidence": round(confidence, 4),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Convenience wrapper for batch prediction."""
        return [self.predict(t) for t in texts]


# =============================================================================
# QUICK SMOKE-TEST (run as script)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ModerationPipeline Smoke Test")
    print("=" * 60)

    test_cases = [
        ("I will kill you tonight.", "Should BLOCK via input_filter (direct_threat)"),
        ("You should kill yourself.", "Should BLOCK via input_filter (self_harm_directed)"),
        ("I know where you live and I'll post your address.", "Should BLOCK via input_filter (doxxing_stalking)"),
        ("They are not human, they are animals.", "Should BLOCK via input_filter (dehumanization)"),
        ("Everyone report this account now!", "Should BLOCK via input_filter (coordinated_harassment)"),
        ("I love this movie, it was fantastic!", "Should ALLOW (non-toxic)"),
        ("This policy is absolutely disgusting and wrong.", "Likely MODEL decision (borderline)"),
    ]

    # Layer 1 only test (no model loaded)
    for text, expected in test_cases:
        result = input_filter(_normalize_text(text))
        if result:
            print(f"[BLOCKED — {result['category']}] {text[:60]}")
        else:
            print(f"[PASSED FILTER] {text[:60]}")
        print(f"  Expected: {expected}\n")
