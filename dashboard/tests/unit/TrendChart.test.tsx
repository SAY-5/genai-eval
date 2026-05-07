import { describe, expect, it, beforeAll } from "vitest";
import { render, screen } from "@testing-library/react";

import { TrendChart } from "@/components/TrendChart";
import type { TrendPoint } from "@/lib/api";

// recharts uses ResponsiveContainer which needs a measurable size in jsdom.
// Stub the layout primitives so the chart actually renders its SVG content.
beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, "clientWidth", {
    configurable: true,
    value: 600,
  });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", {
    configurable: true,
    value: 300,
  });
  // jsdom lacks ResizeObserver, which recharts uses through ResponsiveContainer.
  if (!(globalThis as unknown as { ResizeObserver?: unknown }).ResizeObserver) {
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = class {
      observe(): void {}
      unobserve(): void {}
      disconnect(): void {}
    };
  }
});

const points: TrendPoint[] = [
  {
    run_id: 1,
    model: "fake-large",
    started_at: "2026-05-01T00:00:00Z",
    task: "qa",
    language: "en",
    pass_rate: 0.5,
  },
  {
    run_id: 2,
    model: "fake-large",
    started_at: "2026-05-02T00:00:00Z",
    task: "qa",
    language: "en",
    pass_rate: 0.75,
  },
  {
    run_id: 1,
    model: "fake-large",
    started_at: "2026-05-01T00:00:00Z",
    task: "qa",
    language: "es",
    pass_rate: 1.0,
  },
  {
    run_id: 2,
    model: "fake-large",
    started_at: "2026-05-02T00:00:00Z",
    task: "qa",
    language: "es",
    pass_rate: 1.0,
  },
];

describe("TrendChart", () => {
  it("mounts and exposes its testid container", () => {
    render(<TrendChart points={points} />);
    expect(screen.getByTestId("trend-chart")).toBeTruthy();
  });

  it("mounts the wrapper for non-empty input", () => {
    const { container } = render(<TrendChart points={points} />);
    const wrapper = container.querySelector("[data-testid='trend-chart']");
    expect(wrapper).not.toBeNull();
    // The recharts ResponsiveContainer must be present even though the
    // SVG body may not be measured under jsdom.
    expect(wrapper?.querySelector(".recharts-responsive-container")).not.toBeNull();
  });

  it("renders without crashing when given an empty point list", () => {
    render(<TrendChart points={[]} />);
    expect(screen.getByTestId("trend-chart")).toBeTruthy();
  });
});
