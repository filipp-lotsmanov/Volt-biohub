"""Tests for shared-GPU selection.

Ranking is pure logic over sampled readings, so it is tested against synthetic
`Gpu` records — no GPU, no driver, and no dependence on who else is using the box
at the time the suite runs.
"""

from volt_biohub.gpu import Gpu, format_table, rank, scan


def _g(index, free_gib, util=0.0, cap=8.6, total_gib=48, name="Card"):
    return Gpu(index=index, name=name, total_mib=total_gib * 1024,
               free_mib=int(free_gib * 1024), util_pct=util, compute_cap=cap)


def test_more_free_memory_wins():
    """Free memory dominates: it is what actually kills a run."""
    got = rank([_g(0, 10), _g(1, 40), _g(2, 25)])
    assert [g.index for g in got] == [1, 2, 0]


def test_utilisation_breaks_ties_at_equal_memory():
    """A busy card with the same free memory must lose to an idle one."""
    got = rank([_g(0, 43.0, util=78), _g(1, 43.9, util=0)])
    assert [g.index for g in got] == [1, 0]


def test_utilisation_is_bucketed_not_raw():
    """3% vs 7% is noise, so it must not reorder cards; index breaks the tie."""
    got = rank([_g(3, 40, util=7), _g(1, 40, util=3)])
    assert [g.index for g in got] == [1, 3]


def test_newer_arch_wins_when_otherwise_equal():
    """An Ada card (sm_89) has ~1.7x the A6000's cores; prefer it on a true tie."""
    got = rank([_g(0, 40, cap=8.6), _g(1, 40, cap=8.9)])
    assert [g.index for g in got] == [1, 0]


def test_ranking_is_stable_for_identical_cards():
    got = rank([_g(5, 40), _g(2, 40), _g(7, 40)])
    assert [g.index for g in got] == [2, 5, 7]


def test_unavailable_utilisation_sorts_last_not_idle():
    """`[N/A]` util is parsed as 100 so a wedged card never looks free."""
    busted = _g(0, 40, util=100.0)
    healthy = _g(1, 40, util=0.0)
    assert [g.index for g in rank([busted, healthy])] == [1, 0]


def test_format_table_marks_the_pick():
    table = format_table([_g(0, 10), _g(1, 40)], chosen=[1])
    lines = table.splitlines()
    assert lines[0].split() == ["idx", "name", "free", "GiB", "util", "cc", "usage"]
    # Best-first, so the chosen card is the first data row and carries the marker.
    assert lines[1].startswith("  1") and "<= pick" in lines[1]
    assert "<= pick" not in lines[2]


def test_scan_returns_empty_without_a_driver(monkeypatch):
    """Must degrade to CPU rather than crash — this also runs on the dev box."""
    monkeypatch.setattr("volt_biohub.gpu.shutil.which", lambda _: None)
    assert scan() == []
