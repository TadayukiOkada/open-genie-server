"""Documentation checks that used to be run by hand after every doc edit.

The English and Japanese manuals are maintained as pairs; the failure mode is
always the same — one side gains a section and the other does not, and nobody
notices until a reader follows a link that no longer exists. These run in the
offline suite (no NPU, no network) so CI catches it on the PR.

Fenced code blocks are skipped throughout: a shell comment inside one looks
exactly like a Markdown heading.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^(```+|~~~+)")
HEADING = re.compile(r"^(#{1,6}) +(.*)")


def _markdown_files():
    return sorted(p for p in REPO_ROOT.rglob("*.md") if ".git" not in p.parts)


def _parse(path: Path):
    """(headings, prose) with fenced blocks removed.

    headings is a list of (level, text); prose is everything outside fences,
    which is where a link has to be to actually be a link.
    """
    headings, prose, fence = [], [], None
    for line in path.read_text().splitlines():
        m = FENCE.match(line)
        if m:
            token = m.group(1)[0] * 3
            fence = token if fence is None else (None if token == fence else fence)
            continue
        if fence:
            continue
        prose.append(line)
        m = HEADING.match(line)
        if m:
            headings.append((len(m.group(1)), m.group(2).strip()))
    return headings, "\n".join(prose)


def _slug(heading: str) -> str:
    """GitHub's anchor slug, close enough for link checking: drop code ticks
    and link syntax, lowercase, drop punctuation, spaces to hyphens. CJK
    characters survive, which is what the Japanese docs link to."""
    heading = re.sub(r"`([^`]*)`", r"\1", heading)
    heading = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    heading = re.sub(r"[^\w\s\-　-鿿]", "", heading.lower().strip())
    return heading.replace(" ", "-")


def _anchors(headings) -> set:
    """Every #anchor the file offers, including GitHub's -1, -2 suffixes for
    repeated heading text."""
    counts, anchors = {}, set()
    for _level, text in headings:
        s = _slug(text)
        counts[s] = counts.get(s, 0) + 1
        anchors.add(s if counts[s] == 1 else f"{s}-{counts[s] - 1}")
    return anchors


def _pairs():
    out = []
    for en in _markdown_files():
        if en.name.endswith(".ja.md"):
            continue
        ja = en.with_name(en.name[:-len(".md")] + ".ja.md")
        if ja.exists():
            out.append((en, ja))
    return out


@pytest.mark.parametrize("en,ja", _pairs(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_translated_docs_have_the_same_sections(en, ja):
    """Same number of headings, at the same levels, in the same order — a
    section added on one side only is the failure this catches."""
    en_heads, _ = _parse(en)
    ja_heads, _ = _parse(ja)
    assert [lvl for lvl, _ in en_heads] == [lvl for lvl, _ in ja_heads], (
        f"{en.name} has {len(en_heads)} headings, {ja.name} has {len(ja_heads)}"
        " (or they are nested differently)")


def test_there_is_a_translated_pair_to_check():
    """Guards the parametrization above: a globbing mistake would make every
    pair test vanish and the suite still pass."""
    assert len(_pairs()) >= 8


@pytest.mark.parametrize("path", _markdown_files(),
                         ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_internal_anchors_resolve(path):
    """A renamed heading leaves every in-page link to it pointing at nothing;
    the browser silently does not scroll, so this is invisible in review."""
    headings, prose = _parse(path)
    anchors = _anchors(headings)
    broken = [m.group(1) for m in re.finditer(r"\]\(#([^)]+)\)", prose)
              if m.group(1) not in anchors]
    assert not broken, f"{path.name}: no heading for {broken}"


@pytest.mark.parametrize("path", _markdown_files(),
                         ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_cross_file_anchors_resolve(path):
    """The same check as above, for links that name a file *and* a heading.

    These were the blind spot: the in-page test never sees them because they
    carry a path, and the file-exists test throws the fragment away. A link
    into another page's section is exactly where a heading rename goes
    unnoticed — and where a hand-written Japanese slug goes wrong.
    """
    _headings, prose = _parse(path)
    broken = []
    for m in re.finditer(r"\[[^\]]*\]\(([^)#\s]+)#([^)\s]+)\)", prose):
        target, fragment = m.group(1), m.group(2)
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        other = path.parent / target
        if not other.exists() or other.suffix != ".md":
            continue        # the file-exists test owns that failure
        if fragment not in _anchors(_parse(other)[0]):
            broken.append(f"{target}#{fragment}")
    assert not broken, f"{path.name}: no heading for {broken}"


@pytest.mark.parametrize("path", _markdown_files(),
                         ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_relative_links_point_at_files_that_exist(path):
    """Catches a link left behind by a moved or deleted file."""
    _headings, prose = _parse(path)
    missing = []
    for m in re.finditer(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)", prose):
        target = m.group(1)
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (path.parent / target).exists():
            missing.append(target)
    assert not missing, f"{path.name}: missing {missing}"
