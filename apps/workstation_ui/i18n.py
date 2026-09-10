"""UTF-8 Chinese-first catalog with English fallback."""
import json
from pathlib import Path


class Translator:
    def __init__(self, locale: str = "zh-CN", catalog_directory: Path | None = None):
        directory = catalog_directory or Path(__file__).parent / "locales"
        self.catalogs = {name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
                         for name in ("en-US", "zh-CN")}
        self.locale = locale if locale in self.catalogs else "en-US"

    def __call__(self, key: str, **values) -> str:
        text = self.catalogs[self.locale].get(key, self.catalogs["en-US"].get(key, key))
        return text.format(**values) if values else text
