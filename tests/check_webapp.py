"""Opt-in real browser check: start the built local app, then run this script."""

import csv
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright, expect


ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "runs/task8"
output.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto("http://127.0.0.1:8000", wait_until="networkidle")
    expect(page.get_by_text("Local engine connected")).to_be_visible(timeout=30000)
    expect(page.get_by_role("button", name="Run inspection")).to_be_disabled()
    page.screenshot(path=str(output / "desktop-empty.png"), full_page=True)
    page.get_by_role("button", name="Sample B", exact=True).click()
    expect(page.get_by_role("button", name="Run inspection")).to_be_enabled()
    for model, run in (("patchcore", "patchcore_seed42"), ("lbp", "lbp_seed42"), ("lbp_glcm", "lbp_glcm_seed42")):
        page.locator(f'input[name="model"][value="{model}"]').check()
        with page.expect_response(lambda response: "/api/predict?" in response.url, timeout=120000) as pending:
            page.get_by_role("button", name="Run inspection").click()
        response = pending.value
        assert response.ok, response.text()
        result = response.json()
        with (ROOT / f"runs/{run}/evaluation/clean/predictions.csv").open() as stream:
            saved = next(row for row in csv.DictReader(stream) if row["path"] == "test/cut/000.png")
        np.testing.assert_allclose(result["score"], float(saved["raw_score"]), rtol=1e-5 if model == "patchcore" else 0, atol=1e-5 if model == "patchcore" else 1e-12)
        assert result["threshold"] == float(saved["threshold"])
        assert (result["decision"] == "anomaly") == bool(int(saved["decision"]))
        expect(page.get_by_text("Inspection complete", exact=True)).to_be_visible()
        if model == "patchcore":
            slider = page.get_by_role("slider", name="Original and heatmap comparison")
            slider.focus()
            slider.press("ArrowRight")
            expect(slider).to_have_value("51")
            opacity = page.get_by_role("slider", name="Overlay opacity")
            opacity.fill("35")
            expect(page.locator(".opacity-control output")).to_have_text("35%")
            page.screenshot(path=str(output / "desktop-heatmap.png"), full_page=True)
        else:
            expect(page.get_by_text("Localization unavailable for this model.")).to_be_visible()
        print(f"PASS: {model} browser prediction matches frozen clean result")
    page.get_by_role("button", name="Experiment results").click()
    expect(page.get_by_role("heading", name="Model comparison")).to_be_visible()
    page.get_by_role("button", name="Brightness ×1.2").click()
    expect(page.locator(".metric-panel")).to_have_count(3)
    expect(page.locator(".metric-panel").nth(1)).to_contain_text("100.0%")
    page.screenshot(path=str(output / "desktop-results.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.get_by_role("button", name="Inspection", exact=True).click()
    page.screenshot(path=str(output / "mobile-inspection.png"), full_page=True)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    upload = page.get_by_label("Upload carpet image")
    upload.set_input_files({"name": "broken.png", "mimeType": "image/png", "buffer": b"broken"})
    expect(page.get_by_role("alert")).to_contain_text("Cannot read this image")
    expect(page.get_by_role("button", name="Run inspection")).to_be_disabled()
    upload.set_input_files({"name": "wrong.gif", "mimeType": "image/gif", "buffer": b"GIF89a"})
    expect(page.get_by_role("alert")).to_contain_text("Choose a PNG or JPEG")
    # A wide image exercises overlay alignment instead of just square dataset images.
    wide = BytesIO()
    Image.new("RGB", (600, 180), "gray").save(wide, format="PNG")
    upload.set_input_files({"name": "wide.png", "mimeType": "image/png", "buffer": wide.getvalue()})
    expect(page.get_by_role("button", name="Run inspection")).to_be_enabled()
    page.locator('input[name="model"][value="patchcore"]').check()
    with page.expect_response(lambda response: "/api/predict?" in response.url, timeout=120000):
        page.get_by_role("button", name="Run inspection").click()
    expect(page.get_by_role("slider", name="Original and heatmap comparison")).to_be_visible()
    original = page.locator(".original-image").bounding_box()
    overlay = page.locator(".heatmap-layer").bounding_box()
    assert abs(original["width"] / original["height"] - 600 / 180) < .02
    assert all(abs(original[key] - overlay[key]) < 1 for key in ("x", "y", "width", "height"))
    assert not errors, errors
    browser.close()
print("PASS: heatmap controls, measured results, mobile layout, upload errors, non-square alignment; no browser exceptions")
