"""Japanese lyrics as romaji or kana, for singers who cannot read kanji.

Kanji have several readings each, so neither form can be derived character by
character: a dictionary tokenizer (fugashi with UniDic) picks each word's
reading from context, and the rest is table work. Readings are right for most
lyrics; a poetic reading a lyricist chose on purpose can come out as the
common one.
"""

import logging
import re
import unicodedata
from collections.abc import Callable

_KANA = re.compile(r"[ぁ-ゖァ-ヺ]")
_KANJI = re.compile(r"[㐀-䶿一-鿿豈-﫿々]")
# Stretches of Japanese script. Everything between them (English, digits,
# spaces, punctuation) is kept as written rather than tokenized.
_JAPANESE_RUN = re.compile(
    r"[ぁ-ゟ゠-ヿ㐀-䶿一-鿿豈-﫿々〆ヶ]+"
)

# Hepburn, as karaoke romaji writes it: long vowels spelled out (kyou, riyuu).
# fmt: off
_DIGRAPHS = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo", "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho", "しぇ": "she",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo", "じぇ": "je",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho", "ちぇ": "che",
    "ぢゃ": "ja", "ぢゅ": "ju", "ぢょ": "jo",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo", "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo", "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo", "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    # Loanword spellings.
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo", "ふゅ": "fyu",
    "てぃ": "ti", "でぃ": "di", "とぅ": "tu", "どぅ": "du", "でゅ": "dyu",
    "うぃ": "wi", "うぇ": "we", "うぉ": "wo", "つぁ": "tsa", "つぃ": "tsi", "つぇ": "tse",
    "つぉ": "tso", "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo", "いぇ": "ye",
}
# fmt: on
_SINGLE = dict(
    zip(
        "あいうえおかきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでど"
        "なにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわゐゑをんゔ"
        "ぁぃぅぇぉゃゅょゎ",
        "a i u e o ka ki ku ke ko ga gi gu ge go sa shi su se so za ji zu ze zo "
        "ta chi tsu te to da ji zu de do na ni nu ne no ha hi fu he ho ba bi bu be bo "
        "pa pi pu pe po ma mi mu me mo ya yu yo ra ri ru re ro wa i e wo n vu "
        "a i u e o ya yu yo wa".split(),
    )
)
# Particles read differently from how they are written.
_PARTICLES = {"は": "wa", "へ": "e", "を": "wo"}
# Conjunctive particles that inflect the word before: nite, ikeba, shitatte.
_JOINING_PARTICLES = {"て", "で", "ば", "ちゃ", "じゃ", "たり", "だり", "たって", "だって"}
# Auxiliaries that are words of their own, not inflections: yume naraba.
_STANDALONE_AUXILIARIES = {"だ", "です", "でしょう", "だろう", "なら", "ならば", "らしい", "みたい"}
_INFLECTING = {"動詞", "形容詞", "助動詞"}
_ROMAJI_PUNCTUATION = str.maketrans(
    {"、": ", ", "。": ". ", "「": "", "」": "", "『": "", "』": "", "・": " ", "〜": "~"}
)
# Marks a group already in romaji (a particle), so it is not converted twice.
_DONE = "\0"

_tagger = None


def is_japanese(text: str) -> bool:
    """Whether text is Japanese: kana, which Chinese lacks, settles it."""
    return bool(_KANA.search(text))


def _get_tagger():
    global _tagger
    if _tagger is None:
        import fugashi  # Deferred: loading the dictionary takes a moment and ~50 MB.

        _tagger = fugashi.Tagger()
        logging.info("Japanese dictionary loaded")
    return _tagger


def _words(japanese: str) -> list[tuple[str, str | None, str, str]]:
    """(surface, katakana reading or None, part of speech, its subtype) per word."""
    return [
        (w.surface, w.feature.kana, w.feature.pos1, w.feature.pos2)
        for w in _get_tagger()(japanese)
    ]


def _convert(line: str, japanese: Callable[[str], str], other: Callable[[str], str]) -> str:
    """Convert the Japanese stretches of a line one way and the rest another."""
    line = unicodedata.normalize("NFKC", line)
    out, last = [], 0
    for run in _JAPANESE_RUN.finditer(line):
        out += [other(line[last : run.start()]), japanese(run.group())]
        last = run.end()
    out.append(other(line[last:]))
    return "".join(out)


def to_hiragana(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def to_kana(line: str) -> str:
    """The line with every kanji word spelled out in hiragana; all else as written."""
    return _convert(line, _kana_run, lambda text: text)


def _kana_run(japanese: str) -> str:
    return "".join(
        to_hiragana(reading) if reading and _KANJI.search(surface) else surface
        for surface, reading, _, _ in _words(japanese)
    )


def to_romaji(line: str) -> str:
    """The line in Hepburn romaji, spaced by word the way karaoke romaji is."""
    out = _convert(line, lambda run: f" {_romaji_run(run)} ", lambda t: t.translate(_ROMAJI_PUNCTUATION))
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.!?)\]])", r"\1", out)
    out = re.sub(r"([(\[])\s+", r"\1", out).strip()
    return out[:1].upper() + out[1:]


def _romaji_run(japanese: str) -> str:
    """Romaji for a stretch of Japanese: one word per group of a word and its inflections.

    A group is converted as one, so a small tsu ending one token doubles the
    consonant starting the next (wakatta, not wakata).
    """
    groups: list[str] = []
    previous_pos = ""
    for surface, reading, pos, subtype in _words(japanese):
        kana = to_hiragana(reading or surface)
        last = groups[-1] if groups else None
        joins = (
            last is not None
            and not last.startswith(_DONE)
            and (
                last.endswith("っ")
                or pos == "接尾辞"
                or (pos == "助詞" and subtype == "接続助詞" and surface in _JOINING_PARTICLES)
                or (
                    pos == "助動詞"
                    and surface not in _STANDALONE_AUXILIARIES
                    and previous_pos in _INFLECTING
                )
            )
        )
        if joins:
            groups[-1] += kana
        elif pos == "助詞" and surface in _PARTICLES:
            groups.append(_DONE + _PARTICLES[surface])
        else:
            groups.append(kana)
        previous_pos = pos
    return " ".join(g[1:] if g.startswith(_DONE) else kana_to_romaji(g) for g in groups)


def kana_to_romaji(kana: str) -> str:
    """Hiragana to Hepburn; anything else passes through."""
    out: list[str] = []
    i = 0
    while i < len(kana):
        pair, char = kana[i : i + 2], kana[i]
        if pair in _DIGRAPHS:
            out.append(_DIGRAPHS[pair])
            i += 2
            continue
        if char == "っ":
            # A small tsu doubles the next consonant: kitte, matcha.
            following = kana_to_romaji(kana[i + 1 : i + 3])[:1]
            out.append("t" if following == "c" else following if following.isalpha() else "")
        elif char == "ー":
            # The long-vowel mark repeats the vowel before it.
            vowels = [c for c in "".join(out) if c in "aeiou"]
            out.append(vowels[-1] if vowels else "")
        elif char == "ん":
            # n' before a vowel or y keeps "kan'i" apart from "kani".
            following = kana_to_romaji(kana[i + 1 : i + 2])[:1]
            out.append("n'" if following in ("a", "i", "u", "e", "o", "y") else "n")
        else:
            out.append(_SINGLE.get(char, char))
        i += 1
    return "".join(out)
