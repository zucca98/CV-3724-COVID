"""Rigenera ``docs/technical_analysis.pdf`` (e la versione italiana) dai .md.

Sostituisce il vecchio flusso pandoc/LaTeX con un toolchain pure-Python
basato su ``markdown-pdf`` (nessuna dipendenza LaTeX richiesta). Tiene il
PDF allineato al sorgente Markdown - utile in CI o come comando one-shot
quando si modifica il report.

Uso (dalla root del progetto):
    python scripts/build_report_pdf.py
    python scripts/build_report_pdf.py --src docs/it/analisi_tecnica.md \\
                                       --out docs/it/analisi_tecnica.pdf
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

logger = logging.getLogger("build_report_pdf")

# Regex per le sintassi pandoc-specifiche che ``markdown-pdf`` non gestisce.
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_PANDOC_IMG_ATTR_RE = re.compile(r"\)\{[^}]*\}")  # ){ width=45% } -> )
_NEWPAGE_RE = re.compile(r"^\\newpage\s*$", re.MULTILINE)
_PAGE_MARKER = "\n<!--PAGEBREAK-->\n"
# ![alt](path) - cattura solo immagini con path locali (no http/https/data:).
_IMG_RE = re.compile(r"(!\[[^\]]*\]\()(?!https?:|data:)([^)\s]+)(\))")

_CSS = """
body { font-family: 'DejaVu Sans', Arial, sans-serif; line-height: 1.45;
       font-size: 10.5pt; }
h1 { font-size: 20pt; border-bottom: 2px solid #444; padding-bottom: 4px;
     margin-top: 0; }
h2 { font-size: 15pt; border-bottom: 1px solid #999; padding-bottom: 2px;
     margin-top: 1.2em; }
h3 { font-size: 12pt; margin-top: 1em; }
code, pre { font-family: 'Consolas', 'DejaVu Sans Mono', monospace;
            font-size: 9.5pt; background: #f5f5f5; }
pre { padding: 8px; border-left: 3px solid #888; overflow-x: auto; }
table { border-collapse: collapse; margin: 0.6em 0; }
th, td { border: 1px solid #bbb; padding: 4px 8px; }
th { background: #eee; }
img { max-width: 90%; }
blockquote { border-left: 3px solid #888; padding-left: 10px; color: #555; }
"""


def _rebase_images(md: str, src_dir: Path, archive_root: Path) -> str:
    """Riscrive i path immagine perche' siano relativi a ``archive_root``.

    ``markdown-pdf`` usa internamente ``fitz.Story(archive=...)`` di PyMuPDF
    che si aspetta percorsi **relativi** alla cartella ``archive``: non
    risolve ``..`` ne' accetta path assoluti / URI ``file://``. Per gestire
    sia i sorgenti EN (``docs/technical_analysis.md`` che usa ``figures/...``)
    sia IT (``docs/it/analisi_tecnica.md`` che usa ``../figures/...``)
    riscriviamo ogni path come ``rel_to(archive_root)``.
    """

    def _sub(match: re.Match[str]) -> str:
        prefix, path, suffix = match.group(1), match.group(2), match.group(3)
        resolved = (src_dir / path).resolve()
        try:
            rel = resolved.relative_to(archive_root.resolve())
        except ValueError:
            # File fuori dall'archive: lasciamo invariato (verra' loggato a video).
            logger.warning("image %s is outside archive root %s", resolved, archive_root)
            return match.group(0)
        return f"{prefix}{rel.as_posix()}{suffix}"

    return _IMG_RE.sub(_sub, md)


def _strip_pandoc_syntax(md: str, src_dir: Path, archive_root: Path) -> list[str]:
    """Rimuove frontmatter / attributi pandoc e divide in sezioni di pagina."""
    md = _FRONTMATTER_RE.sub("", md, count=1)
    md = _PANDOC_IMG_ATTR_RE.sub(")", md)
    md = _NEWPAGE_RE.sub(_PAGE_MARKER, md)
    md = _rebase_images(md, src_dir, archive_root)
    parts = [p.strip() for p in md.split(_PAGE_MARKER)]
    return [p for p in parts if p]


def build(src: Path, out: Path, *, toc_level: int = 3,
          archive_root: Path | None = None) -> None:
    if not src.exists():
        raise FileNotFoundError(src)

    # Default: ``docs/`` se la sorgente vive sotto ``docs/`` (regge sia
    # ``docs/foo.md`` sia ``docs/it/foo.md``); altrimenti la cartella del file.
    if archive_root is None:
        try:
            archive_root = next(p for p in src.resolve().parents if p.name == "docs")
        except StopIteration:
            archive_root = src.parent

    sections = _strip_pandoc_syntax(src.read_text(encoding="utf-8"),
                                    src.parent, archive_root)
    logger.info("source=%s archive=%s sections=%d", src, archive_root, len(sections))

    pdf = MarkdownPdf(toc_level=toc_level)
    for body in sections:
        pdf.add_section(Section(body, root=str(archive_root)), user_css=_CSS)

    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(str(out))
    logger.info("wrote %s (%.1f KB)", out, out.stat().st_size / 1024)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, default=Path("docs/technical_analysis.md"))
    p.add_argument("--out", type=Path, default=None,
                   help="default: stesso path di --src con estensione .pdf")
    p.add_argument("--toc-level", type=int, default=3)
    p.add_argument("--archive-root", type=Path, default=None,
                   help="root per i path immagine (default: il primo ancestor 'docs/')")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    out = args.out if args.out is not None else args.src.with_suffix(".pdf")
    build(args.src, out, toc_level=args.toc_level, archive_root=args.archive_root)


if __name__ == "__main__":
    main()
