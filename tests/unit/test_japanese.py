"""Tests for turning Japanese lyrics into romaji and kana."""

import pytest

from pikaraoke.lib import japanese


@pytest.mark.parametrize(
    "line, romaji",
    [
        ("二人だけの空が広がる夜に", "Futari dake no sora ga hirogaru yoru ni"),
        # A small tsu ending one word doubles the next consonant.
        ("その一言で全てが分かった", "Sono hitokoto de subete ga wakatta"),
        ("フェンス越しに重なっていた", "Fensugoshi ni kasanatte ita"),
        # Particles are read wa and wo, not ha and wo-as-o.
        ("君は僕に似ている", "Kimi wa boku ni nite iru"),
        ("強くなれる理由を知った", "Tsuyoku nareru riyuu wo shitta"),
        # Auxiliaries that are words of their own stay apart.
        ("夢ならばどれほどよかったでしょう", "Yume naraba dore hodo yokatta deshou"),
        ("どうしたって！", "Dou shitatte!"),
        ("それもそっか", "Sore mo sokka"),
    ],
)
def test_romaji_reads_lyrics_as_karaoke_romaji(line, romaji):
    assert japanese.to_romaji(line) == romaji


def test_english_in_a_line_is_kept_as_written():
    assert (
        japanese.to_romaji("夜の匂いに (I'll spend all thirty nights)")
        == "Yoru no nioi ni (I'll spend all thirty nights)"
    )
    assert japanese.to_kana("ドキドキしてる my heart") == "ドキドキしてる my heart"


def test_kana_spells_out_kanji_and_keeps_the_rest():
    assert japanese.to_kana("強くなれる理由を知った 僕を連れて進め") == (
        "つよくなれるりゆうをしった ぼくをつれてすすめ"
    )


def test_quotes_drop_from_romaji():
    assert japanese.to_romaji("「さよなら」だけだった") == "Sayonara dake datta"


@pytest.mark.parametrize(
    "kana, romaji",
    [
        ("きって", "kitte"),
        ("まっちゃ", "matcha"),
        ("らーめん", "raamen"),
        ("かんい", "kan'i"),
        ("しゃしん", "shashin"),
        ("ふぁん", "fan"),
    ],
)
def test_kana_to_romaji_spelling(kana, romaji):
    assert japanese.kana_to_romaji(kana) == romaji


def test_japanese_is_told_from_chinese_by_its_kana():
    assert japanese.is_japanese("夜に駆ける")
    assert not japanese.is_japanese("月亮代表我的心")
    assert not japanese.is_japanese("Lemon")
