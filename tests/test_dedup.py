import synth  # noqa: F401
from synth import deduplicate as dd

A = ("Once upon a time there was a little girl named Lily. She loved to play with her red ball in the park. "
     "One day the ball rolled into the pond and Lily was sad. A kind duck pushed the ball back to her.")
B_NEAR = A.replace("red ball", "blue ball").replace("kind duck", "nice duck")
C_OTHER = ("Tom had a big truck. He drove it to the farm every morning. The cows liked the noise. "
           "One night it rained and the truck got stuck in the mud, so the horse helped him pull it out.")


def test_exact_and_near_duplicates_removed_keep_first():
    keep, why = dd.dedup_within([A, A.upper(), B_NEAR, C_OTHER])
    assert keep == [True, False, False, True]
    assert why[1] == "exact_duplicate" and why[2] == "near_duplicate"


def test_jaccard_exact_values():
    sa, sb, sc = dd.shingles(A), dd.shingles(B_NEAR), dd.shingles(C_OTHER)
    assert dd.jaccard(sa, sa) == 1.0
    assert dd.jaccard(sa, sb) >= 0.5 > dd.jaccard(sa, sc)


def test_index_finds_near_copy_only():
    idx = dd.build_index([A, C_OTHER])
    sims = dd.max_sim_against([B_NEAR, "Something entirely different about a dragon and a cake in space."], idx)
    assert sims[0] >= 0.5 and sims[1] < 0.5


def test_ngram_contamination():
    g = dd.ngram_set([A])
    assert dd.shares_ngram("x " + A[:120], g)
    assert not dd.shares_ngram(C_OTHER, g)


def test_deterministic_signature():
    s = dd.shingles(A)
    assert (dd.signature(s) == dd.signature(dd.shingles(A))).all()
