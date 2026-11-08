"""项目数据与抽取结果路径。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def data_dir() -> str:
    return str(PROJECT_ROOT)

def cleaned_dir() -> str:
    return str(PROJECT_ROOT / "cleaned_data")

def output_dir() -> str:
    path = PROJECT_ROOT / "data" / "extracted"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
