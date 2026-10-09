from plagai.text import count_words, normalize, sentence_windows, split_sentences


def test_normalize_removes_zero_width_and_folds_mixed_script_homoglyphs():
    attacked = "Thе m​odel wаs trained."  # Cyrillic е and а inside Latin words
    assert normalize(attacked) == "The model was trained."


def test_normalize_keeps_genuine_greek_and_cyrillic():
    assert normalize("Привет мир and αβγ") == "Привет мир and αβγ"


def test_split_sentences_offsets_and_abbreviations():
    text = "Dr. Smith went home. He slept! Did he dream?\nNew paragraph here"
    spans = split_sentences(text)
    assert [s.text for s in spans] == ["Dr. Smith went home.", "He slept!", "Did he dream?", "New paragraph here"]
    for s in spans:
        assert text[s.start:s.end] == s.text


def test_windows_cover_all_sentences():
    sents = split_sentences(" ".join(f"Sentence number {i}." for i in range(10)))
    wins = sentence_windows(sents, 4, 2)
    assert wins[0] == (0, 4) and wins[-1][1] == 10


def test_count_words():
    assert count_words("It's a test, isn't it?") == 5
