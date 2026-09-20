"""Unit tests for dataset loaders parsing raw format files."""

from pathlib import Path

from backend.app.dataset.loaders.fakenewsnet import FakeNewsNetLoader
from backend.app.dataset.loaders.liar import LiarDatasetLoader


def test_liar_loader_parsing(tmp_path: Path):
    """Verify LiarDatasetLoader correctly maps 14-column TSV format."""
    tsv_content = "1001.json\tfalse\tThe state budget was completely depleted yesterday.\tfinance\tJohn Politician\tGovernor\tTexas\trepublican\t1\t2\t3\t4\t5\tspeech at rally\n"
    test_file = tmp_path / "test_liar.tsv"
    test_file.write_text(tsv_content, encoding="utf-8")

    loader = LiarDatasetLoader()
    records = loader.load_records(test_file)

    assert len(records) == 1
    rec = records[0]
    assert rec.dataset_name == "liar"
    assert rec.original_id == "1001.json"
    assert rec.original_label == "false"
    assert rec.text == "The state budget was completely depleted yesterday."
    assert rec.author == "John Politician"
    assert rec.metadata["speaker"] == "John Politician"
    assert rec.metadata["credit_history"]["pants_on_fire"] == "5"


def test_fakenewsnet_loader_parsing(tmp_path: Path):
    """Verify FakeNewsNetLoader parses CSV records."""
    csv_content = (
        "id,news_url,title,tweet_ids\n"
        "politifact_123,https://example.com/fake-news,Celebrity confirms alien encounter,12345\n"
    )
    test_file = tmp_path / "politifact_fake.csv"
    test_file.write_text(csv_content, encoding="utf-8")

    loader = FakeNewsNetLoader()
    records = loader.load_records(test_file)

    assert len(records) == 1
    rec = records[0]
    assert rec.dataset_name == "fakenewsnet"
    assert rec.original_id == "politifact_123"
    assert rec.original_label == "fake"
    assert rec.text == "Celebrity confirms alien encounter"
    assert rec.source_url == "https://example.com/fake-news"
