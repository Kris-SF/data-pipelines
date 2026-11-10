"""Build factor-lab.html: inject data.json (daily log returns, 1e-5 units) into template.html.

    python build.py            # writes factor-lab.html next to this file
"""
from pathlib import Path

here = Path(__file__).parent
html = (here / "template.html").read_text().replace("__DATA__", (here / "data.json").read_text())
(here / "factor-lab.html").write_text(html)
print("wrote", here / "factor-lab.html")
