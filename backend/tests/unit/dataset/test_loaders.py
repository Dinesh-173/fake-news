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


def test_liar_loader_directory_loading(tmp_path: Path):
    """Verify LiarDatasetLoader correctly loads and tags a directory containing train, valid, and test TSVs."""
    liar_dir = tmp_path / "liar_benchmark"
    liar_dir.mkdir()

    row_train = "1.json\ttrue\tStatement 1\tfinance\tSpeaker\tJob\tState\tParty\t0\t0\t0\t0\t0\tcontext\n"
    row_val = "2.json\tfalse\tStatement 2\tfinance\tSpeaker\tJob\tState\tParty\t0\t0\t0\t0\t0\tcontext\n"
    row_test = "3.json\thalf-true\tStatement 3\tfinance\tSpeaker\tJob\tState\tParty\t0\t0\t0\t0\t0\tcontext\n"

    (liar_dir / "train.tsv").write_text(row_train, encoding="utf-8")
    (liar_dir / "valid.tsv").write_text(row_val, encoding="utf-8")
    (liar_dir / "test.tsv").write_text(row_test, encoding="utf-8")

    loader = LiarDatasetLoader()
    records = loader.load_records(liar_dir)

    assert len(records) == 3
    rec_train = next(r for r in records if r.original_id == "1.json")
    rec_val = next(r for r in records if r.original_id == "2.json")
    rec_test = next(r for r in records if r.original_id == "3.json")

    assert rec_train.split == "train"
    assert rec_train.original_split == "train"
    assert rec_val.split == "val"
    assert rec_val.original_split == "val"
    assert rec_test.split == "test"
    assert rec_test.original_split == "test"


def test_liar_loader_authentic_corpus_record_counts():
    """Verify the authentic local LIAR corpus has exact record counts: train=10,269, val=1,284, test=1,283, total=12,836."""
    raw_path = Path("data/raw/liar")
    if not raw_path.exists():
        return

    loader = LiarDatasetLoader()
    records = loader.load_records(raw_path)

    assert len(records) == 12836
    train_count = sum(1 for r in records if r.split == "train")
    val_count = sum(1 for r in records if r.split == "val")
    test_count = sum(1 for r in records if r.split == "test")

    assert train_count == 10269
    assert val_count == 1284
    assert test_count == 1283


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
