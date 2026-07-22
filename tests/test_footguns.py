"""Tests for the curated external-API footgun registry."""
from sift.core.footguns import detect_footguns, format_footgun_notes


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


def test_unsupported_language_returns_nothing():
    # .lua has no language mapping at all (unlike .ts/.java/.rb, which are now supported below).
    diff = _diff("local x = queryset[offset:end];")
    assert detect_footguns("app/x.lua", diff, "QuerySet order_by objects") == []


def test_supported_non_python_language_with_no_matching_pattern_returns_nothing():
    # .ts IS a supported language now, but this content matches none of its footgun patterns.
    diff = _diff("const x = queryset[offset:end];")
    assert detect_footguns("app/x.ts", diff, "QuerySet order_by objects") == []


def test_no_added_lines_returns_nothing():
    assert detect_footguns("src/x.py", "", "QuerySet") == []


def test_format_footgun_notes_empty_and_nonempty():
    assert format_footgun_notes([]) == ""
    block = format_footgun_notes(["note one", "note two"])
    assert "- note one" in block and "- note two" in block
    assert block.startswith("Known runtime footguns")


# ---- Java ----

def test_java_optional_get_fires():
    diff = _diff("        String key = credential.getOptionalKey().get();")
    content = "Optional<String> getOptionalKey() { return Optional.empty(); }\n"
    notes = detect_footguns("src/main/java/Foo.java", diff, content)
    assert len(notes) == 1
    assert "NoSuchElementException" in notes[0]


def test_java_map_get_with_key_ignored():
    # Map.get(key) / List.get(i) take an argument — not the bare Optional.get() footgun.
    diff = _diff("        String v = myMap.get(\"key\");")
    content = "Map<String, String> myMap;\n"
    assert detect_footguns("src/main/java/Foo.java", diff, content) == []


def test_java_picocli_exit_fires():
    diff = _diff("        return commandLine.exit(exitCode);")
    content = "import picocli.CommandLine;\nCommandLine commandLine;\n"
    notes = detect_footguns("src/main/java/Cli.java", diff, content)
    assert len(notes) == 1
    assert "System.exit" in notes[0]


def test_java_exit_without_picocli_context_ignored():
    diff = _diff("        return commandLine.exit(exitCode);")
    assert detect_footguns("src/main/java/Cli.java", diff, "no relevant import here") == []


# ---- Ruby ----

def test_ruby_kernel_open_ssrf_fires():
    diff = _diff("    data = open(params[:url]).read")
    notes = detect_footguns("app/models/importer.rb", diff, "url = params[:url]\n")
    assert len(notes) == 1
    assert "SSRF" in notes[0]


def test_ruby_file_open_not_flagged():
    diff = _diff("    data = File.open(path).read")
    assert detect_footguns("app/models/importer.rb", diff, "url = path\n") == []


def test_ruby_x_frame_options_allowall_fires():
    diff = _diff('    response.headers["X-Frame-Options"] = "ALLOWALL"')
    notes = detect_footguns("app/controllers/embed_controller.rb", diff, "")
    assert len(notes) == 1
    assert "clickjacking" in notes[0]


def test_ruby_unanchored_domain_regex_fires():
    diff = _diff('    email =~ /@(#{domains})/')
    notes = detect_footguns("app/models/blocked_email.rb", diff, "")
    assert len(notes) == 1
    assert "DOMAIN SUFFIX" in notes[0]


def test_ruby_anchored_domain_regex_not_flagged():
    diff = _diff('    email =~ /@(#{domains})\\z/')
    assert detect_footguns("app/models/blocked_email.rb", diff, "") == []


def test_ruby_string_literal_append_mutation_fires():
    diff = _diff("    host = '.' << website_host")
    notes = detect_footguns("app/serializers/site_serializer.rb", diff, "")
    assert len(notes) == 1
    assert "FrozenError" in notes[0]


def test_ruby_variable_append_not_flagged():
    diff = _diff("    host = prefix << website_host")
    assert detect_footguns("app/serializers/site_serializer.rb", diff, "") == []


def test_ruby_include_predicate_missing_question_mark_fires():
    diff = _diff("  def include_website_name")
    content = "class SiteSerializer < ActiveModel::Serializer\nend\n"
    notes = detect_footguns("app/serializers/site_serializer.rb", diff, content)
    assert len(notes) == 1
    assert "include_<attr>?" in notes[0]


def test_ruby_include_predicate_with_question_mark_not_flagged():
    diff = _diff("  def include_website_name?")
    content = "class SiteSerializer < ActiveModel::Serializer\nend\n"
    assert detect_footguns("app/serializers/site_serializer.rb", diff, content) == []


def test_ruby_duplicate_def_fires():
    diff = _diff("  def downsize(dimensions)")
    content = (
        "def downsize(max_width, max_height)\n  # first definition\nend\n\n"
        "def downsize(dimensions)\n  # second definition overrides the first\nend\n"
    )
    notes = detect_footguns("lib/image_optim.rb", diff, content)
    assert any("overrides" in n for n in notes)


def test_ruby_single_def_not_flagged_as_duplicate():
    diff = _diff("  def downsize(dimensions)")
    content = "def downsize(dimensions)\n  # only definition\nend\n"
    notes = detect_footguns("lib/image_optim.rb", diff, content)
    assert not any("overrides" in n for n in notes)


# ---- TypeScript / JavaScript ----

def test_ts_foreach_async_fires():
    diff = _diff("  items.forEach(async (item) => { await remove(item); });")
    notes = detect_footguns("src/booking/cleanup.ts", diff, "")
    assert len(notes) == 1
    assert "fire-and-forget" in notes[0]


def test_ts_map_async_not_flagged():
    # map+async is fine when the caller awaits Promise.all — only bare forEach is unconditionally buggy.
    diff = _diff("  await Promise.all(items.map(async (item) => { await remove(item); }));")
    assert detect_footguns("src/booking/cleanup.ts", diff, "") == []


def test_ts_dayjs_triple_equals_fires():
    diff = _diff("  const same = dayjs(date.start).add(offset, 'minutes') === dayjs(date.end);")
    notes = detect_footguns("src/lib/availability.ts", diff, "")
    assert len(notes) == 1
    assert "isSame" in notes[0]


def test_ts_dayjs_is_same_not_flagged():
    diff = _diff("  const same = dayjs(date.start).isSame(dayjs(date.end));")
    assert detect_footguns("src/lib/availability.ts", diff, "") == []


def test_ts_backup_code_indexof_case_sensitivity_fires():
    diff = _diff("  const valid = backupCodes.indexOf(candidate) !== -1;")
    notes = detect_footguns("src/lib/twoFactor/BackupCode.ts", diff, "")
    assert len(notes) == 1
    assert "case-sensitive" in notes[0]


def test_indexof_without_security_context_not_flagged():
    diff = _diff("  const found = names.indexOf(candidate) !== -1;")
    assert detect_footguns("src/lib/util.ts", diff, "") == []


def test_js_extension_inherits_typescript_footgun():
    # Plain .js (e.g. an older client-side embed script) gets the same JS-family footguns as .ts.
    diff = _diff("  items.forEach(async (item) => { await remove(item); });")
    notes = detect_footguns("public/embed.js", diff, "")
    assert len(notes) == 1
    assert "fire-and-forget" in notes[0]
