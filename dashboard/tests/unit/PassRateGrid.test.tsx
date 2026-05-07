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
});
