import { expect, test } from "@playwright/test";

test.describe("dashboard", () => {
  test("home shows recent runs and links to detail", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("h1")).toContainText("genai-eval");
    const card = page.getByTestId("run-card");
    await expect(card).toBeVisible();
    await card.click();
    await expect(page).toHaveURL(/\/runs\/\d+$/);
  });

  test("run detail renders the pass-rate grid", async ({ page }) => {
    await page.goto("/runs/1");
    await expect(page.getByTestId("pass-rate-grid")).toBeVisible();
    await expect(page.getByTestId("trend-chart")).toBeVisible();
  });

  test("unknown route returns 404", async ({ page }) => {
    const resp = await page.goto("/does-not-exist");
    expect(resp?.status()).toBe(404);
  });
});
