"""Claude policy installer (spec sections 15.2 and 23).

The installer touches a file the user owns and may have written by hand, so
these tests are mostly about what it must *not* destroy.
"""

from __future__ import annotations

from pathlib import Path

from workerq.claude_policy import (
    END_MARKER,
    POLICY_BODY,
    START_MARKER,
    count_blocks,
    find_block,
    install_policy,
    install_safe_launcher,
    policy_block,
    policy_status,
    remove_block,
    remove_policy,
    safe_launcher_status,
    upsert_block,
)

EXISTING = """# My instructions

Always write tests first.

## Style
Prefer clarity over cleverness.
"""


def test_append_to_new_file(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    result = install_policy(target)
    assert result["created"] is True
    assert result["changed"] is True
    assert result["backup"] is None

    text = target.read_text(encoding="utf-8")
    assert START_MARKER in text and END_MARKER in text
    assert "workerq submit --project" in text


def test_creates_parent_directory(tmp_path: Path):
    target = tmp_path / "nested" / "deeper" / "CLAUDE.md"
    install_policy(target)
    assert target.exists()


def test_preserves_existing_file(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    target.write_text(EXISTING, encoding="utf-8")
    install_policy(target)

    text = target.read_text(encoding="utf-8")
    assert "Always write tests first." in text
    assert "Prefer clarity over cleverness." in text
    assert "# My instructions" in text
    assert START_MARKER in text


def test_backup_made_before_modifying_an_existing_file(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    target.write_text(EXISTING, encoding="utf-8")
    result = install_policy(target)

    assert result["backup"] is not None
    backup = Path(result["backup"])
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == EXISTING


def test_install_is_idempotent(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    install_policy(target)
    first = target.read_text(encoding="utf-8")

    second_result = install_policy(target)
    assert second_result["changed"] is False
    assert target.read_text(encoding="utf-8") == first
    assert count_blocks(target.read_text(encoding="utf-8")) == 1


def test_repeated_installs_never_duplicate(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    for _ in range(5):
        install_policy(target)
    assert count_blocks(target.read_text(encoding="utf-8")) == 1


def test_updates_an_outdated_block(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    stale = f"# Mine\n\n{START_MARKER}\nOLD POLICY TEXT\n{END_MARKER}\n\n## After\nkeep me\n"
    target.write_text(stale, encoding="utf-8")

    result = install_policy(target)
    text = target.read_text(encoding="utf-8")

    assert result["changed"] is True
    assert "OLD POLICY TEXT" not in text
    assert "workerq submit --project" in text
    assert "# Mine" in text
    assert "keep me" in text
    assert count_blocks(text) == 1


def test_content_after_the_block_is_preserved(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    target.write_text(
        f"BEFORE\n\n{START_MARKER}\nold\n{END_MARKER}\n\nAFTER\n", encoding="utf-8"
    )
    install_policy(target)
    text = target.read_text(encoding="utf-8")
    assert text.index("BEFORE") < text.index(START_MARKER) < text.index("AFTER")


def test_duplicate_blocks_collapse_to_one(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    target.write_text(
        f"{START_MARKER}\na\n{END_MARKER}\n\nmiddle\n\n{START_MARKER}\nb\n{END_MARKER}\n",
        encoding="utf-8",
    )
    install_policy(target)
    text = target.read_text(encoding="utf-8")
    assert count_blocks(text) == 1
    assert "middle" in text


def test_remove_only_the_gpuq_block(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    target.write_text(EXISTING, encoding="utf-8")
    install_policy(target)

    result = remove_policy(target)
    text = target.read_text(encoding="utf-8")

    assert result["changed"] is True
    assert START_MARKER not in text
    assert END_MARKER not in text
    assert "Always write tests first." in text
    assert "Prefer clarity over cleverness." in text
    assert "# My instructions" in text


def test_remove_backs_up_first(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    install_policy(target)
    result = remove_policy(target)
    assert result["backup"] and Path(result["backup"]).exists()


def test_remove_when_absent_is_a_no_op(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    target.write_text(EXISTING, encoding="utf-8")
    result = remove_policy(target)
    assert result["changed"] is False
    assert target.read_text(encoding="utf-8") == EXISTING


def test_remove_when_no_file(tmp_path: Path):
    result = remove_policy(tmp_path / "nothing.md")
    assert result["changed"] is False


def test_status_reports_states(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"

    status = policy_status(target)
    assert status["exists"] is False and status["installed"] is False

    target.write_text(EXISTING, encoding="utf-8")
    status = policy_status(target)
    assert status["exists"] is True and status["installed"] is False

    install_policy(target)
    status = policy_status(target)
    assert status["installed"] is True and status["current"] is True and status["blocks"] == 1


def test_status_detects_outdated_block(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    target.write_text(f"{START_MARKER}\nstale\n{END_MARKER}\n", encoding="utf-8")
    status = policy_status(target)
    assert status["installed"] is True
    assert status["current"] is False


# --- pure block helpers ---------------------------------------------------


def test_upsert_into_empty_string():
    assert upsert_block("", policy_block()).strip().startswith(START_MARKER)


def test_find_block_returns_none_when_absent():
    assert find_block("no markers here") is None
    assert find_block(START_MARKER + " unterminated") is None


def test_remove_block_on_text_without_markers():
    assert remove_block("plain text") == "plain text"


def test_policy_text_matches_the_spec():
    body = POLICY_BODY
    for required in (
        "## Heavy Workload Policy (GPU, RAM and CPU)",
        "NEVER directly launch a command",
        "workerq submit --project <project> --priority normal -- <command> <args...>",
        "workerq status",
        "workerq cancel <job_id>",
        "Do not bypass `workerq`",
        "Small CPU-only commands",
        # The policy must teach declaring a footprint, or admission control is
        # guessing and jobs get admitted that should have waited.
        "--ram 24 --vram 12 --cpus 4",
        "workerq top",
        "workerq report",
        # Preemption is destructive for non-resumable work, so the policy
        # must teach both the lever and the caveat.
        "workerq bump <job_id> critical",
        "workerq wait <job_id>",
        "--preemptible",
        "Do not resubmit it",
    ):
        assert required in body


def test_unicode_content_is_preserved(tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Notes café 日本\n\nkeep\n", encoding="utf-8")
    install_policy(target)
    assert "café 日本" in target.read_text(encoding="utf-8")


# --- safe launcher --------------------------------------------------------


def test_safe_launcher_install_and_status(tmp_path: Path):
    assert safe_launcher_status(tmp_path)["installed"] is False
    result = install_safe_launcher(tmp_path)
    assert result["paths"]
    status = safe_launcher_status(tmp_path)
    assert status["installed"] is True

    body = Path(result["paths"][0]).read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" in body


def test_safe_launcher_is_not_enabled_globally(tmp_path: Path):
    """It must be an explicit opt-in, never wired in automatically."""
    result = install_safe_launcher(tmp_path)
    assert "NOT enabled globally" in result["message"]
