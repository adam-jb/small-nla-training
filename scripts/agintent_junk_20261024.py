"""Foreign-script junk in an NLA note (agentic-intent GRPO, 24 Oct 2026).

A note is junk if any quoted span ("...", “...”) or any single word mixes Latin letters with letters of another script
(CJK, kana, Hangul, Cyrillic, Arabic, Indic, Georgian, Thai...), e.g. "your静的", "Ob젝つき", "Nഗveen". A pure non-Latin
quote or word (a real Russian or Chinese phrase) is not junk. Context-free on purpose: the trainer only sees vectors.
GRPO526 held-out notes (1,020): see the check in the 24 Oct handoff. Used as a GRPO penalty (arm J) and as a metric.
"""
import re
import unicodedata

QUOTE_RE = re.compile(r'"([^"\n]{1,400})"|“([^”\n]{1,400})”')
WORD_RE = re.compile(r'\w+')


def _scripts(s):
    lat = oth = 0
    for ch in s:
        if ch.isalpha():
            if ch.isascii() or 'LATIN' in unicodedata.name(ch, ''):
                lat += 1
            else:
                oth += 1
    return lat, oth


def is_junk(text):
    for m in QUOTE_RE.finditer(text):
        lat, oth = _scripts(m.group(1) or m.group(2))
        if lat >= 2 and oth >= 1:
            return True
    for w in WORD_RE.findall(text):
        lat, oth = _scripts(w)
        if lat and oth:
            return True
    return False
