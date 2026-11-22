"""M1 phase 2: the dictionary of surface forms, wildcards (k-gram and permuterm), edit distance, spelling suggestions, Soundex, and
wildcard / party-name queries through the search engine. Small hand-made corpora; nothing here reads the real corpus."""

from __future__ import annotations

import random
import re

import pytest

pytest.importorskip("nltk")
try:
    from m1_index import dictionary as D
    from m1_index.index import IndexBuilder, memoised_tokenizer
    from m1_index.query_parser import PhraseNode, WildcardNode, WildcardPhraseNode, parse_query
    from m1_index.searcher import RankedSearchEngine
    from m1_index.tokenizer import surface_tokens, tokenize
except (LookupError, RuntimeError, ImportError) as exc:  # NLTK stop words missing, for example
    pytest.skip(f"M1 cannot be imported here: {exc}", allow_module_level=True)

from common.schema import Judgment

CFG = dict(D.DEFAULTS)


# ------------------------------------------------------------------------------------------------ pure helpers
def test_kgrams_follow_the_textbook():
    assert D.kgrams("castle") == {"$ca", "cas", "ast", "stl", "tle", "le$"}
    assert D.kgrams("a") == {"$a$"}
    assert D.pattern_kgrams("murd*") == {"$mu", "mur", "urd"}
    assert D.pattern_kgrams("*cide") == {"cid", "ide", "de$"}
    assert D.pattern_kgrams("re*ve") == {"$re", "ve$"}
    assert D.pattern_kgrams("a*") == set()  # "$a" is shorter than k


def test_pattern_regex_is_anchored_and_escapes_everything_but_the_star():
    rx = D.pattern_regex("re*ve")
    assert rx.match("relive") and rx.match("reve") and not rx.match("relived") and not rx.match("xrelive")
    assert D.pattern_regex("120-b*").match("120-b") and not D.pattern_regex("a.b").match("axb")


@pytest.mark.parametrize("a,b,plain,swap", [
    ("kitten", "sitting", 3, 3), ("", "abc", 3, 3), ("abc", "abc", 0, 0), ("flaw", "lawn", 2, 2),
    ("culpabel", "culpable", 2, 1), ("ca", "ac", 2, 1), ("murdr", "murder", 1, 1), ("ca", "abc", 3, 3),
])
def test_levenshtein_known_values(a, b, plain, swap):
    assert D.levenshtein(a, b) == plain == D.levenshtein(b, a)
    assert D.levenshtein(a, b, transpositions=True) == swap


def _osa_reference(a: str, b: str) -> int:
    """Optimal string alignment distance with the full matrix, written independently of the two-row version."""
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        m[i][0] = i
    for j in range(len(b) + 1):
        m[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            m[i][j] = min(m[i - 1][j] + 1, m[i][j - 1] + 1, m[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                m[i][j] = min(m[i][j], m[i - 2][j - 2] + 1)
    return m[-1][-1]


def test_levenshtein_cutoff_and_transpositions_agree_with_a_full_matrix():
    rng = random.Random(5)
    for _ in range(2000):
        a = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 7)))
        b = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 7)))
        full = _osa_reference(a, b)
        assert D.levenshtein(a, b, transpositions=True) == full
        for cap in (0, 1, 2):
            got = D.levenshtein(a, b, cap, transpositions=True)
            assert got == (full if full <= cap else cap + 1), (a, b, cap)


@pytest.mark.parametrize("name,code", [
    ("Robert", "R163"), ("Rupert", "R163"), ("Ashcraft", "A261"), ("Tymczak", "T522"), ("Pfister", "P236"), ("Lee", "L000"),
    ("Mohammed", "M530"), ("Muhammad", "M530"), ("Mohammad", "M530"), ("Mohd", "M530"), ("Mohd.", "M530"),
    ("Shivaji", "S120"), ("Sivaji", "S120"), ("Shivaji Rao", "S120 R000"), ("", ""), ("123", ""),
])
def test_soundex(name, code):
    assert D.soundex(name) == code


def test_soundex_index_matches_every_word_of_the_name():
    sx = D.SoundexIndex([("d1", "Mohd. Arif v. State of U.P."), ("d2", "Muhammad Rafiq v. Union of India"),
                         ("d3", "Sivaji Rao v. State"), ("d4", "State of Punjab v. Gurmit Singh")])
    assert sx.match("Mohammed") == {"d1", "d2"}
    assert sx.match("Shivaji") == {"d3"}
    assert sx.match("Mohammed Rafeeq") == {"d2"}
    assert sx.match("State") == set()  # title words that are not party names are not indexed
    assert sx.match("") == set()


def test_surface_tokens_pair_every_index_term_with_its_word():
    texts = ["The appellants were MURDERED under Section 302 IPC and 120-B; the [2025] 1 S.C.R. 1 citation.",
             "Cr.P.C. section 482: quashing of the F.I.R. was refused; intention, intentionally, intended.", "", "of the and"]
    for text in texts:
        pairs = surface_tokens(text)
        assert [t for _, t in pairs] == tokenize(text)
        for word, term in pairs:
            assert word == word.lower() and tokenize(word) == [term]


# ------------------------------------------------------------------------------------------------ the dictionary
WORDS = {"murder": 50, "murdered": 20, "murderer": 8, "murders": 15, "murky": 3, "homicide": 30, "suicide": 12, "decide": 40,
         "conspiracy": 25, "consipracy": 1, "culpable": 30, "culpabil": 4, "intention": 40, "intent": 20, "intentional": 9,
         "dowry": 18, "death": 60, "seth": 2, "bail": 70, "sale": 10, "302": 45, "302-a": 2, "498-a": 12, "ipc": 80, "prosecution": 55}


@pytest.fixture(scope="module")
def small_dict() -> D.SurfaceDictionary:
    return D.SurfaceDictionary.from_counts(WORDS)


def test_wildcards_match_a_brute_force_scan(small_dict):
    for pattern in ("murd*", "*cide", "cons*acy", "int*n*", "*", "m*", "302*", "*-a", "x*", "murder", "nothing*"):
        want = sorted(w for w in WORDS if re.fullmatch(pattern.replace("*", ".*"), w))
        try:
            got = sorted(small_dict.words[i] for i in small_dict.wildcard_ids(pattern))
        except ValueError:
            assert pattern == "*"  # nothing to look up: refused, never a scan of the whole vocabulary
            continue
        assert got == want, pattern


def test_kgram_candidates_are_post_filtered(small_dict):
    stats: dict = {}
    small_dict.wildcard_ids("cons*acy", stats)
    assert stats["matches"] == 2 and stats["candidates"] >= stats["matches"]
    stats = {}
    assert small_dict.wildcard_ids("mur*er", stats) == [small_dict.word_id("murder"), small_dict.word_id("murderer")]


def test_expand_wildcard_orders_by_df_and_limits(small_dict):
    assert small_dict.expand_wildcard("murd*") == ["murder", "murdered", "murders", "murderer"]
    assert small_dict.expand_wildcard("murd*", limit=2) == ["murder", "murdered"]
    assert small_dict.wildcard_terms("murd*") == ["murder"]  # every form stems to the same index term
    assert set(small_dict.wildcard_terms("intent*")) == {tokenize("intention")[0]}


def test_permuterm_agrees_with_the_kgram_index_and_is_larger():
    rng = random.Random(3)
    words = sorted({"".join(rng.choice("abcde") for _ in range(rng.randint(1, 8))) for _ in range(400)})
    d = D.SurfaceDictionary.from_counts({w: 1 for w in words})
    perm = D.PermutermIndex(d.words)
    patterns = ["ab*", "*de", "a*e", "*b*", "a*b*c", "abc", "c*a*", "*ee*"] + [
        "".join(rng.choice("abcde*") for _ in range(rng.randint(2, 5))) for _ in range(200)]
    for p in patterns:
        if len(p.replace("*", "")) < 1:
            continue
        p = re.sub(r"\*+", "*", p)
        try:
            k = d.wildcard_ids(p)
        except ValueError:
            continue
        assert perm.lookup(p) == k, p
    sizes = D.size_comparison(d, perm)
    assert sizes["permuterm_rotations"] == sum(len(w) + 1 for w in words)
    assert sizes["ratio"] > 1


def test_candidates_rank_by_distance_then_df(small_dict):
    cands = small_dict.candidates("culpabel", 2, 0.2, transpositions=True)
    assert cands[0][:3] == ("culpable", 1, 30)  # swap = 1 edit; ties with culpabil (1 edit) go to the higher df
    assert ("culpabil", 1, 4) == cands[1][:3]
    plain = small_dict.candidates("culpabel", 2, 0.2, transpositions=False)
    assert plain[0][0] == "culpabil"  # without transpositions the rarer word wins on distance


def test_suggest_corrects_misspellings_and_never_legal_tokens(small_dict):
    s = lambda q: small_dict.suggest(q, cfg=CFG)  # noqa: E731
    assert s("culpabel homicide") == "culpable homicide"
    assert s("Murdr") == "Murder"
    assert s("CONSPIRCY") == "CONSPIRACY"
    assert s("dowry deth") == "dowry death"
    assert s("murder") is None and s("") is None
    assert s("section 302 IPC") is None  # numbers and abbreviations are never touched
    assert s("302-x murdr") == "302-x murder"
    assert s("murd* AND homicid*") is None  # wildcards are not corrected
    assert s("culpabel /s murdr") == "culpable /s murder"
    assert s('"culpabel homicide" AND NOT murdr') == '"culpable homicide" AND NOT murder'


def test_suggest_rare_words_need_a_much_more_frequent_candidate(small_dict):
    cfg = dict(CFG, suggest_rare_df=3, suggest_min_df=3)
    # "consipracy" occurs in 1 judgment, "conspiracy" in 25
    assert small_dict.suggest("consipracy", cfg=dict(cfg, suggest_df_ratio=20)) == "conspiracy"
    assert small_dict.suggest("consipracy", cfg=dict(cfg, suggest_df_ratio=30)) is None
    # a word above the rare threshold is checked only when the query finds few judgments: "murders" (15) vs "murder" (50)
    assert small_dict.suggest("murders", cfg=dict(cfg, suggest_df_ratio=3)) is None
    assert small_dict.suggest("murders", hit_count=lambda q: 100, cfg=dict(cfg, suggest_df_ratio=3)) is None
    assert small_dict.suggest("murders", hit_count=lambda q: 0, cfg=dict(cfg, suggest_df_ratio=3)) == "murder"
    assert small_dict.suggest("murders", hit_count=lambda q: 0, cfg=dict(cfg, suggest_df_ratio=4)) is None


def test_dictionary_save_and_load_round_trip(tmp_path, small_dict):
    small_dict.save(tmp_path)
    back = D.SurfaceDictionary.load(tmp_path)
    assert back.words == small_dict.words and list(back.df) == list(small_dict.df)
    assert back.grams == small_dict.grams and back.g_terms == small_dict.g_terms
    assert back.expand_wildcard("*cide") == small_dict.expand_wildcard("*cide")
    with pytest.raises(FileNotFoundError):
        D.SurfaceDictionary.load(tmp_path / "missing")


# ------------------------------------------------------------------------------------------------ parser
def test_parser_records_wildcards():
    assert parse_query("murd*") == WildcardNode("murd*")
    assert parse_query("MURD**") == WildcardNode("murd*")
    node = parse_query('"common intent*"')
    assert isinstance(node, WildcardPhraseNode) and node.parts == ["common", WildcardNode("intent*")]
    assert parse_query('"the intent*"') == WildcardNode("intent*")
    assert isinstance(parse_query('"common intention"'), PhraseNode)
    for bad in ("*", "a*", "**", "mu$*"):
        with pytest.raises(ValueError):
            parse_query(bad)


# ------------------------------------------------------------------------------------------------ the engine
DOCS = [
    ("d1", "Mohd. Arif v. State", {"headnote": "Murder by common intention.", "holding": "The accused murdered the victim with common intention."}),
    ("d2", "Muhammad Rafiq v. State", {"headnote": "Culpable homicide not amounting to murder.", "facts": "A suicide note was found."}),
    ("d3", "Sivaji Rao v. State", {"headnote": "Dowry death.", "holding": "Common intentions were not shown; the murderer was acquitted."}),
    ("d4", "State v. Gurmit Singh", {"headnote": "Criminal conspiracy and cheating.", "holding": "Conspiracy proved; the court must decide."}),
    ("d5", "Ram Lal v. State", {"facts": "The common object and intention of the murderers.", "arguments": "Intentional homicide."}),
]


@pytest.fixture(scope="module")
def engine() -> RankedSearchEngine:
    builder, counter = IndexBuilder(), D.SurfaceCounter()
    with memoised_tokenizer():
        for doc_id, title, zones in DOCS:
            builder.add(Judgment(doc_id=doc_id, title=title, date="2020-01-01", bench_size=2, judges=[], reporter_citations=[],
                                 zones=zones, text=" ".join(zones.values())))
            counter.add(zones.values())
    return RankedSearchEngine(index=builder.finish(), dictionary=D.SurfaceDictionary.from_counts(counter.counts))


def ids(hits) -> set[str]:
    return {h.doc_id for h in hits}


def test_wildcard_query_is_the_or_of_its_expansions(engine):
    assert ids(engine.search("murd*")) == {"d1", "d2", "d3", "d5"}
    assert ids(engine.search("murd*")) == ids(engine.search("murder OR murdered OR murderer OR murderers"))
    assert ids(engine.search("*cide")) == {"d2", "d4", "d5"}  # homicide, suicide, decide: impossible on stems (homicid)
    assert ids(engine.search("cons*acy AND cheat*")) == {"d4"}
    assert engine.search("zzz*") == []


def test_wildcard_scores_equal_those_of_the_expanded_terms(engine):
    a = engine.search("murd*")
    b = engine.search("murder OR murdered OR murderer OR murderers")
    assert [(h.doc_id, round(h.rel, 9)) for h in a] == [(h.doc_id, round(h.rel, 9)) for h in b]


def test_wildcard_phrase_and_proximity(engine):
    assert ids(engine.search('"common intent*"')) == {"d1", "d3"}  # d5 has "common object and intention": not a phrase
    assert ids(engine.search('"common intent*" /s murd*')) == {"d1", "d3"}
    assert ids(engine.search('"common intent*"')) == ids(engine.search('"common intention" OR "common intentions"'))


def test_naive_and_optimised_evaluation_agree_on_wildcards(engine):
    for q in ("murd*", "*cide AND NOT murd*", '"common intent*"', '"common intent*" /3 murd*', "cons*acy OR dow*"):
        ast = parse_query(q)
        assert engine.evaluate(ast, optimise=True)[0] == engine.evaluate(parse_query(q), optimise=False)[0], q


def test_skips_do_not_change_results(engine):
    for q in ("murder AND intention", '"common intention"', "homicide AND NOT murder", "murd* AND common"):
        engine.skips = False
        plain = engine.search(q)
        engine.skips = True
        assert [(h.doc_id, h.rel) for h in engine.search(q)] == [(h.doc_id, h.rel) for h in plain], q


def test_party_filter_uses_soundex(engine):
    assert ids(engine.search("murder OR homicide", filters={"party": "Mohammed"})) == {"d1", "d2"}
    assert ids(engine.search("dowry", filters={"party": "Shivaji"})) == {"d3"}
    assert engine.search("dowry", filters={"party": "Mohammed"}) == []
    assert ids(engine.search("murder", filters={"party": "Mohammed", "year": 2020})) == {"d1", "d2"}
