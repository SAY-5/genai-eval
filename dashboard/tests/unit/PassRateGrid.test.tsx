import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { PassRateGrid } from "@/components/PassRateGrid";

describe("PassRateGrid", () => {
  it("renders one cell per (task, language) pair from props", () => {
    render(
      <PassRateGrid
        cells={[
          {
            task: "qa",
            language: "en",
            n: 3,
            pass_rate: 0.66,
            mean_cost_usd: 0,
            p95_latency_ms: 0,
            errors: 0,
          },
          {
            task: "qa",
            language: "es",
            n: 3,
            pass_rate: 1.0,
            mean_cost_usd: 0,
            p95_latency_ms: 0,
            errors: 0,
          },
        ]}
      />,
    );
    const grid = screen.getByTestId("pass-rate-grid");
    expect(grid).toBeTruthy();
    // 66% formatted
    expect(grid.textContent).toContain("66%");
    expect(grid.textContent).toContain("100%");
  });

  it("renders an empty placeholder when no cell exists for a task/lang slot", () => {
    render(<PassRateGrid cells={[]} />);
    const grid = screen.getByTestId("pass-rate-grid");
    expect(grid).toBeTruthy();
  });

  it("colors cells by pass-rate threshold", () => {
    const { container } = render(
      <PassRateGrid
        cells={[
          {
            task: "qa",
            language: "en",
            n: 10,
            pass_rate: 0.9,
            mean_cost_usd: 0,
            p95_latency_ms: 0,
            errors: 0,
          },
          {
            task: "qa",
            language: "es",
            n: 10,
            pass_rate: 0.7,
            mean_cost_usd: 0,
            p95_latency_ms: 0,
            errors: 0,
          },
          {
            task: "qa",
            language: "ja",
            n: 10,
            pass_rate: 0.2,
            mean_cost_usd: 0,
            p95_latency_ms: 0,
            errors: 0,
          },
        ]}
      />,
    );
    const cells = Array.from(container.querySelectorAll<HTMLElement>("[data-pass-rate]"));
    const findClasses = (rate: number): string =>
      cells.find((el) => el.getAttribute("data-pass-rate") === String(rate))?.className ?? "";
    expect(findClasses(0.9)).toContain("bg-emerald-700");
    expect(findClasses(0.7)).toContain("bg-amber-700");
    expect(findClasses(0.2)).toContain("bg-rose-800");
  });

  it("only renders columns for languages present in the data", () => {
    const { container } = render(
      <PassRateGrid
        cells={[
          {
            task: "qa",
            language: "en",
            n: 1,
            pass_rate: 1,
            mean_cost_usd: 0,
            p95_latency_ms: 0,
            errors: 0,
          },
        ]}
      />,
    );
    const headers = Array.from(container.querySelectorAll("thead th")).map(
      (th) => th.textContent ?? "",
    );
    expect(headers).toContain("en");
    expect(headers).not.toContain("ja");
    expect(headers).not.toContain("py");
  });

  it("renders n and errors metadata in the cell tooltip", () => {
    const { container } = render(
      <PassRateGrid
        cells={[
          {
            task: "qa",
            language: "en",
            n: 42,
            pass_rate: 0.5,
            mean_cost_usd: 0,
            p95_latency_ms: 0,
            errors: 3,
          },
        ]}
      />,
    );
    const cell = container.querySelector<HTMLElement>("[data-pass-rate]");
    expect(cell?.getAttribute("title")).toBe("n=42, errors=3");
  });
});
