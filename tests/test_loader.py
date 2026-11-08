"""Tests for reading a four-sheet input workbook."""
import pandas as pd
import pytest

from src.config import settings
from src.preprocessing.loader import DataLoader


@pytest.fixture
def workbook(tmp_path, monkeypatch):
    path = tmp_path / "news_with_tags_train.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet in range(4):
            pd.DataFrame([{ "article_file_id": f"article-{sheet}", "content": f"text-{sheet}" }]).to_excel(writer, sheet_name=f"sheet{sheet}", index=False)
    monkeypatch.setattr(settings, "data_raw_dir", str(tmp_path))
    return path


@pytest.mark.parametrize("loader,expected", [
    (DataLoader.load_tagged_company_data, "article-0"),
    (DataLoader.load_tagged_industry_data, "article-1"),
    (DataLoader.load_company_duplication, "article-2"),
    (DataLoader.load_industry_duplication, "article-3"),
])
def test_load_workbook_sheet(workbook, loader, expected):
    assert loader().iloc[0]["article_file_id"] == expected


@pytest.mark.parametrize("value,expected", [("ID_A, ID_B", ["ID_A", "ID_B"]), ("ID_A", ["ID_A"]), ("", [])])
def test_parse_duplication_id(value, expected):
    assert DataLoader.parse_duplication_id(value) == expected
