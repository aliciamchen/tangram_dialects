"""
The content words of a description.

The preregistration defines a description's content words for the
description properties (H3c) and the content-word overlap check: lowercased
alphabetic sequences, with NLTK's English stopwords and one-character words
removed and no stemming. This module is the one definition. The measures
(preprocessing/compute_measures.py) and the descriptive word codings of the
figures (report/figures/_common.py) both import it, so a word counts as a
content word in a figure exactly when it does in a measure.

    from preprocessing.words import content_word_tokens, content_words
    content_word_tokens("the ice skater, the skater")   # ["ice", "skater", "skater"]
    content_words("the ice skater, the skater")         # {"ice", "skater"}

Tests: preprocessing/test_compute_measures.py (the Jaccard and
description-property tests, which pin the rule through the measures)
"""

from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    """NLTK's English stopword list, downloaded on first use."""
    import nltk

    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as corpus

    return frozenset(corpus.words("english"))


def content_word_tokens(text) -> list[str]:
    """The content words of a description in order, repeats kept; empty for a missing description."""
    if not isinstance(text, str):
        return []
    stop = stopwords()
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in stop and len(w) > 1]


def content_words(text) -> set[str]:
    """The distinct content words of a description."""
    return set(content_word_tokens(text))
