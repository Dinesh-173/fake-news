"""Unit tests for the central DatasetProfiler orchestrator."""

from backend.app.dataset.analysis.profiler import DatasetProfiler
from backend.app.dataset.schemas import CanonicalNewsRecord


def _make_dummy_records() -> list[CanonicalNewsRecord]:
    rows = [
        ("1", "true", "LIKELY_TRUE", "The unemployment rate dropped to 3.5 percent.", "Bureau of Labor", "democrat"),
        ("2", "mostly-true", "LIKELY_TRUE", "Clean energy jobs grew by 8 percent this year.", "DOE Press", "democrat"),
        ("3", "false", "LIKELY_FALSE", "Aliens landed in Roswell and signed a treaty.", "Anonymous Pundit", "independent"),
        ("4", "pants-fire", "LIKELY_FALSE", "The ocean has completely evaporated into space.", "Conspiracy Blog", "republican"),
        ("5", "half-true", "MISLEADING_OR_PARTIALLY_SUPPORTED", "Tax cuts increased deficit by 1 trillion dollars.", "Senator Bob", "republican"),
    ]
    records = []
    for r_id, orig, norm, text, src, party in rows:
        records.append(
            CanonicalNewsRecord(
                record_id=f"rec_{r_id}",
                dataset_name="synthetic_liar_fixture",
                original_id=r_id,
                text=text,
                original_label=orig,
                normalized_label=norm,
                author=src,
                source_name_raw=src,
                source_name_normalized=src,
                metadata={"party_affiliation": party},
            )
        )
    return records


def test_dataset_profiler_execution(tmp_path):
    records = _make_dummy_records()
    profiler = DatasetProfiler(random_seed=42, git_commit="test_commit_hash")
    profile = profiler.profile_dataset(
        dataset_name="synthetic_fixture",
        records=records,
        dataset_version="v0.0.1",
    )

    assert profile.dataset_name == "synthetic_fixture"
    assert profile.total_records == 5
    assert profile.environment_provenance.git_commit == "test_commit_hash"
    assert profile.environment_provenance.random_seed == 42
    assert profile.label_analysis.total_records == 5
    assert profile.text_analysis.total_records == 5
    assert profile.source_analysis.unique_sources == 5
    assert profile.statistical_tests.total_tests_conducted == 3
    assert "label_completeness" in profile.readiness_summary

    # Test export
    out_dir = tmp_path / "profiler_output"
    paths = profiler.export_profile(profile, out_dir)
    assert len(paths) == 10
    for _name, p in paths.items():
        assert p.exists()
        assert p.stat().st_size > 0
