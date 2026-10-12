"""Browser QA for the gym: load the default character, generate a motion, screenshot frames.

Run against a live `kimodo-turbo gym` server:
  PLAYWRIGHT_BROWSERS_PATH=<scratch>/browsers uv run --with playwright python tests/gym_qa.py [url] [prompt] [model.glb]
Screenshots and a summary go to out/gym/qa/.
"""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

TOOL_ROOT = Path(__file__).resolve().parents[1]
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "A person runs forward at a steady pace."
MODEL = sys.argv[3] if len(sys.argv) > 3 else None  # optional .glb uploaded through the file chooser
TAG = Path(MODEL).stem if MODEL else "default"


def main() -> None:
    out = TOOL_ROOT / "out" / "gym" / "qa" / TAG
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader"])
        page = browser.new_page(viewport={"width": 1400, "height": 860})
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(URL)
        page.wait_for_function("window.gym && window.gym.model !== null", timeout=60_000)
        page.wait_for_selector("#generate:not([disabled])", timeout=120_000)
        if MODEL:
            page.set_input_files("#file", MODEL)
            page.wait_for_function(f"document.getElementById('model-name').textContent === {json.dumps(Path(MODEL).name)}", timeout=120_000)
            page.wait_for_timeout(500)
        page.screenshot(path=out / "00-loaded.png")
        page.fill("#segments textarea", PROMPT)
        page.click("#generate")
        page.wait_for_function("window.gym.current !== null", timeout=180_000)
        page.click("#play")  # pause
        summary = {"gen_status": page.inner_text("#gen-status"), "model_rig": page.inner_text("#model-rig"),
                   "mapped": page.evaluate("[...window.gym.model.mapping].map(([b, j]) => `${b.name}->${j}`)")}
        for k, fraction in enumerate([0.1, 0.35, 0.6, 0.85]):
            page.eval_on_selector("#scrub", f"el => {{ el.value = {fraction}; el.dispatchEvent(new Event('input')); }}")
            page.wait_for_timeout(700)
            page.screenshot(path=out / f"{k + 1:02d}-t{fraction}.png")
        page.check("#skeleton")
        page.wait_for_timeout(700)
        page.screenshot(path=out / "05-skeleton.png")
        summary["metric"] = page.inner_text("#metric")
        summary["errors"] = errors
        browser.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
