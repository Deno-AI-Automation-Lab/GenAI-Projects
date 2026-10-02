from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://www.cricbuzz.com/")

    selector = "span.font-medium.text-cbTxtPrim.truncate"
    page.wait_for_selector(selector)

    rows = page.locator(selector).evaluate_all("""
        elements => elements
            .filter(element => element.innerText.trim())
            .map(element => element.parentElement.innerText.trim())
    """)

    for row in rows:
        print(" ".join(row.split()))

    page.screenshot(path="score.png")
    print("Saved score.png")
    browser.close()