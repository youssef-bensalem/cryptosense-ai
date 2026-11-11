import type { ReactNode } from "react";
import type { ApiError } from "../api";
import { percent } from "../format";
import type { RangeName } from "../types";

export function Card({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${className}`}>
      {title && <h3 className="card-title">{title}</h3>}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

/** A signed change with an arrow, so direction never depends on color alone. */
export function Delta({ value, digits = 1 }: { value: number | null | undefined; digits?: number }) {
  if (value === null || value === undefined || !Number.isFinite(value)) return <span className="muted">—</span>;
  const direction = value > 0 ? "up" : value < 0 ? "down" : "flat";
  const arrow = value > 0 ? "▲" : value < 0 ? "▼" : "•";
  return (
    <span className={`delta delta-${direction}`}>
      <span aria-hidden="true">{arrow}</span> {percent(value, digits, true)}
    </span>
  );
}

export function Notice({ kind = "info", title, children }: { kind?: "info" | "warning"; title?: string; children: ReactNode }) {
  return (
    <div className={`notice notice-${kind}`} role={kind === "warning" ? "alert" : "note"}>
      <span className="notice-icon" aria-hidden="true">{kind === "warning" ? "!" : "i"}</span>
      <div>
        {title && <strong className="notice-title">{title}</strong>}
        <div>{children}</div>
      </div>
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return <p className="loading" role="status">{label}</p>;
}

export function ErrorBox({ error }: { error: ApiError }) {
  return (
    <Notice kind="warning" title="This could not be loaded">
      {error.message}
    </Notice>
  );
}

export function Segmented<T extends string>({ label, options, value, onChange }: {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className={option.value === value ? "active" : ""}
          aria-pressed={option.value === value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

const RANGE_LABELS: Record<RangeName, string> = { "1m": "1 month", "3m": "3 months", "1y": "1 year", all: "All" };

export function RangeControl({ value, onChange }: { value: RangeName; onChange: (value: RangeName) => void }) {
  const options = (Object.keys(RANGE_LABELS) as RangeName[]).map((name) => ({ value: name, label: RANGE_LABELS[name] }));
  return <Segmented label="Chart range" options={options} value={value} onChange={onChange} />;
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "accent" | "warning" }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}
