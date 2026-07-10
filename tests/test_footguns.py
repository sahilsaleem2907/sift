"""Tests for the curated external-API footgun registry."""
from src.core.footguns import detect_footguns, format_footgun_notes


def _diff(*added: str) -> str:
    return "\n".join(["+++ b/x.py", "@@ -1 +1 @@"] + [f"+{ln}" for ln in added])


def test_django_negative_slice_fires_in_orm_context():
    diff = _diff("        results = queryset[offset:offset + limit]")
    content = "from django.db.models import QuerySet\nqueryset = Model.objects.order_by('-datetime')\n"
    notes = detect_footguns("src/api/paginator.py", diff, content)
    assert len(notes) == 1
    assert "Negative indexing" in notes[0] or "negative" in notes[0].lower()


def test_django_slice_ignored_without_orm_context():
    # same slice shape, but no QuerySet/ORM context → plain list slicing, no note
    diff = _diff("        head = items[start:end]")
    notes = detect_footguns("src/util/text.py", diff, "items = [1, 2, 3]\n")
    assert notes == []


def test_django_note_not_fired_on_prefix_slice():
    # `[:-1]` / `[:5]` have an empty start → not flagged even in ORM context
    diff = _diff("        trimmed = queryset[:-1]")
    content = "qs = Model.objects.all()\n"
    assert detect_footguns("src/api/x.py", diff, content) == []


def test_spawn_process_isinstance_fires():
    diff = _diff("            if isinstance(proc, multiprocessing.Process):")
    content = "import multiprocessing\nctx = multiprocessing.get_context('spawn')\n"
    notes = detect_footguns("src/spans/flusher.py", diff, content)
    assert len(notes) == 1
    assert "SpawnProcess" in notes[0]


def test_spawn_isinstance_ignored_without_spawn_context():
    diff = _diff("            if isinstance(proc, multiprocessing.Process):")
    content = "import multiprocessing\n"  # no spawn
    assert detect_footguns("src/x.py", diff, content) == []


def test_datetime_json_fires():
    diff = _diff("    return json.dumps({'queued': self.queued})")
    content = "from datetime import datetime\nself.queued = datetime.now()\n"
    notes = detect_footguns("src/services/source.py", diff, content)
    assert len(notes) == 1
    assert "JSON" in notes[0]


def test_non_python_file_returns_nothing():
    diff = _diff("const x = queryset[offset:end];")
    assert detect_footguns("app/x.ts", diff, "QuerySet order_by objects") == []


def test_no_added_lines_returns_nothing():
    assert detect_footguns("src/x.py", "", "QuerySet") == []


def test_format_footgun_notes_empty_and_nonempty():
    assert format_footgun_notes([]) == ""
    block = format_footgun_notes(["note one", "note two"])
    assert "- note one" in block and "- note two" in block
    assert block.startswith("Known runtime footguns")
