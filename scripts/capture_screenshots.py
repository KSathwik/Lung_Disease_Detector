"""Capture documentation screenshots of the running application with the installed Microsoft Edge (Playwright).

    python scripts/capture_screenshots.py --url http://127.0.0.1:8001 --image backend/static/_e2e_sample3.png

Requires a running backend that serves the production frontend build (docs/deployment.md) and `pip install playwright`
(no browser download: it drives the installed Edge via channel="msedge"). Writes docs/figures/screenshots/*.png.
The uploaded image should be a test-split radiograph; the screenshots show real model output, not mock data.
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figures" / "screenshots"


def _crop(page, heading: str, path: Path, until: str = None) -> None:
    """Screenshot the card that contains `heading` (and, if given, everything down to the card containing `until`)."""
    js = """([h, u]) => {
      const card = t => { const el = [...document.querySelectorAll('p,span,div,h2,h3')]
          .find(e => e.childElementCount === 0 && e.textContent.trim().toLowerCase() === t.toLowerCase());
        let c = el; while (c && c.parentElement && !(getComputedStyle(c).borderRadius !== '0px' && c.offsetWidth > 400)) c = c.parentElement;
        return c.getBoundingClientRect(); };
      const a = card(h), b = u ? card(u) : a;
      return {x: a.left - 8, y: a.top + window.scrollY - 8, width: a.width + 16, height: (b.bottom - a.top) + 16};
    }"""
    box = page.evaluate(js, [heading, until])
    page.screenshot(path=str(path), clip=box, full_page=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--image", type=Path, required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        page.goto(args.url + "/", wait_until="networkidle")
        page.screenshot(path=str(OUT / "01_analyze_upload.png"), full_page=True)
        page.set_input_files("input[type=file]", str(args.image))
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "02_analyze_selected.png"), full_page=True)
        page.get_by_role("button", name="Analyze Image").click()
        page.get_by_text("Model Attention Visualization", exact=False).first.wait_for(timeout=180_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "03_analyze_result.png"), full_page=True)
        page.get_by_role("button", name="View Explanation").click()
        page.wait_for_timeout(2000)
        page.screenshot(path=str(OUT / "04_gradcam.png"), full_page=True)
        # Tight crops for slides (full-page captures repeat the fixed sidebar mid-page)
        _crop(page, "Predicted Condition", OUT / "07_result_card.png", "Probability Distribution")
        _crop(page, "Model Attention Visualization", OUT / "08_gradcam_card.png")
        page.goto(args.url + "/metrics", wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "05_model_performance.png"), full_page=True)
        page.goto(args.url + "/history", wait_until="networkidle")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "06_history.png"), full_page=True)
        browser.close()
    for f in sorted(OUT.glob("*.png")):
        print("wrote", f.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
