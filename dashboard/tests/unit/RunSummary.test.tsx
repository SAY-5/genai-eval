import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { RunSummary } from "@/components/RunSummary";

describe("RunSummary", () => {
  it("displays the model, pass rate, and item count", () => {
    render(
      <RunSummary
        run={{
          id: 7,
          provider: "fake",
          model: "fake-large",
          started_at: "2026-05-02T10:00:00Z",
          finished_at: "2026-05-02T10:00:05Z",
          status: "complete",
          overall_pass_rate: 0.75,
          n_total: 24,
          n_errors: 0,
        }}
      />,
    );
    expect(screen.getByText("fake-large")).toBeTruthy();
    expect(screen.getByText("75%")).toBeTruthy();
    expect(screen.getByText(/n=24/)).toBeTruthy();
  });
});
