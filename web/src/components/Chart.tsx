import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";
import { date, dateTime } from "../format";

export interface ChartSeries {
  key: string;
  label: string;
  /** A CSS color, normally one of the --series-* variables. */
  color: string;
}

/** Color the first series' line by a category column (for example the trend label of each day). */
export interface ColorBy {
  key: string;
  categories: { value: string; label: string; color: string }[];
}

type Point = { t: string } & Record<string, number | string | null>;

interface ChartProps {
  /** Describes the chart for screen readers. */
  label: string;
  points: Point[];
  series: ChartSeries[];
  kind?: "line" | "bar";
  height?: number;
  log?: boolean;
  times?: boolean;
  colorBy?: ColorBy;
  formatY: (value: number) => string;
  formatValue?: (value: number) => string;
}

const MARGIN = { top: 12, right: 16, bottom: 28, left: 56 };
const TABLE_ROWS = 400;

function useWidth(): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(element);
    setWidth(Math.floor(element.getBoundingClientRect().width));
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

/** The smallest round step (1, 2, 2.5 or 5 times a power of ten) that gives at most `count` + 1 ticks. */
function niceStep(span: number, count: number): number {
  const power = Math.pow(10, Math.floor(Math.log10(span / Math.max(count, 1))));
  for (const multiple of [1, 2, 2.5, 5, 10]) {
    if (span / (multiple * power) <= count + 1) return multiple * power;
  }
  return 10 * power;
}

function linearTicks(min: number, max: number, count: number): number[] {
  if (!(max > min)) return [min];
  const step = niceStep(max - min, count);
  const ticks: number[] = [];
  for (let value = Math.ceil(min / step) * step; value <= max + step * 1e-9; value += step) {
    ticks.push(Math.abs(value) < step * 1e-9 ? 0 : value);
  }
  return ticks;
}

function logTicks(min: number, max: number): number[] {
  // Try coarse round numbers first; if the range is narrow, allow finer ones.
  for (const multiples of [[1], [1, 2, 5], [1, 1.5, 2, 3, 5, 7]]) {
    const ticks: number[] = [];
    for (let power = Math.floor(Math.log10(min)); power <= Math.ceil(Math.log10(max)); power++) {
      for (const multiple of multiples) {
        const value = multiple * Math.pow(10, power);
        if (value >= min && value <= max) ticks.push(value);
      }
    }
    if (ticks.length >= 3 && ticks.length <= 7) return ticks;
    if (ticks.length > 7) return ticks.filter((_, index) => index % Math.ceil(ticks.length / 6) === 0);
  }
  return linearTicks(min, max, 5);
}

function value(point: Point, key: string): number | null {
  const raw = point[key];
  return typeof raw === "number" && Number.isFinite(raw) ? raw : null;
}

export function Chart({
  label, points, series, kind = "line", height = 260, log = false, times = false, colorBy, formatY, formatValue,
}: ChartProps) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState<number | null>(null);
  const clipId = useId();
  const show = formatValue ?? formatY;

  const geometry = useMemo(() => {
    const innerWidth = Math.max(width - MARGIN.left - MARGIN.right, 10);
    const innerHeight = height - MARGIN.top - MARGIN.bottom;
    let min = Infinity;
    let max = -Infinity;
    for (const point of points) {
      for (const item of series) {
        const v = value(point, item.key);
        if (v === null || (log && v <= 0)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (!Number.isFinite(min)) {
      min = 0;
      max = 1;
    }
    if (kind === "bar") min = Math.min(min, 0);
    if (log) {
      const pad = Math.pow(max / min, 0.04) || 1.02;
      min /= pad;
      max *= pad;
    } else if (kind !== "bar") {
      const pad = (max - min) * 0.06 || Math.abs(max) * 0.05 || 1;
      min -= pad;
      max += pad;
    } else {
      max += (max - min) * 0.06 || 1;
    }
    const scale = (v: number) => (log ? Math.log(v) : v);
    const y = (v: number) => MARGIN.top + innerHeight * (1 - (scale(v) - scale(min)) / (scale(max) - scale(min)));
    const slot = innerWidth / Math.max(points.length, 1);
    const x = (index: number) =>
      kind === "bar" || points.length === 1
        ? MARGIN.left + slot * (index + 0.5)
        : MARGIN.left + (innerWidth * index) / (points.length - 1);
    const yTicks = log ? logTicks(min, max) : linearTicks(min, max, Math.max(Math.floor(innerHeight / 44), 2));
    const tickCount = Math.max(Math.min(Math.floor(innerWidth / 120), points.length), 1);
    const xTicks: number[] = [];
    if (points.length > 0) {
      for (let i = 0; i < tickCount; i++) {
        xTicks.push(Math.round(((i + 0.5) * points.length) / tickCount - 0.5));
      }
    }
    return { innerWidth, innerHeight, min, max, x, y, slot, yTicks, xTicks };
  }, [width, height, points, series, log, kind]);

  const { innerWidth, innerHeight, x, y, slot, yTicks, xTicks, min } = geometry;

  const spanDays = points.length > 1
    ? (new Date(points[points.length - 1].t).getTime() - new Date(points[0].t).getTime()) / 86_400_000
    : 0;
  const tickLabel = (index: number) => {
    const t = points[index].t;
    if (times) return dateTime(t).replace(" UTC", "");
    if (spanDays > 500) return date(t).split(" ").slice(1).join(" ");
    return date(t, false);
  };

  const paths = useMemo(() => {
    if (kind === "bar" || width === 0) return [];
    return series.map((item) => {
      let d = "";
      let pen = false;
      points.forEach((point, index) => {
        const v = value(point, item.key);
        if (v === null || (log && v <= 0)) {
          pen = false;
          return;
        }
        d += `${pen ? "L" : "M"}${x(index).toFixed(1)},${y(v).toFixed(1)}`;
        pen = true;
      });
      return d;
    });
  }, [points, series, kind, log, width, x, y]);

  const categoryPaths = useMemo(() => {
    if (!colorBy || kind === "bar" || width === 0 || series.length === 0) return null;
    const key = series[0].key;
    return colorBy.categories.map((category) => {
      let d = "";
      for (let index = 1; index < points.length; index++) {
        if (points[index][colorBy.key] !== category.value) continue;
        const from = value(points[index - 1], key);
        const to = value(points[index], key);
        if (from === null || to === null || (log && (from <= 0 || to <= 0))) continue;
        d += `M${x(index - 1).toFixed(1)},${y(from).toFixed(1)}L${x(index).toFixed(1)},${y(to).toFixed(1)}`;
      }
      return d;
    });
  }, [colorBy, points, series, kind, log, width, x, y]);

  const lastIndex = (key: string) => {
    for (let index = points.length - 1; index >= 0; index--) {
      if (value(points[index], key) !== null) return index;
    }
    return -1;
  };

  const onMove = (event: PointerEvent<SVGSVGElement>) => {
    if (points.length === 0) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const position = event.clientX - bounds.left - MARGIN.left;
    const index = kind === "bar" || points.length === 1
      ? Math.floor(position / slot)
      : Math.round((position / innerWidth) * (points.length - 1));
    setHover(Math.min(Math.max(index, 0), points.length - 1));
  };

  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (points.length === 0) return;
    const step = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
    if (event.key === "Escape") setHover(null);
    if (step === 0) return;
    event.preventDefault();
    setHover((current) => Math.min(Math.max((current ?? points.length - 1) + step, 0), points.length - 1));
  };

  const legend = colorBy
    ? colorBy.categories.map((category) => ({ key: category.value, label: category.label, color: category.color }))
    : series.length > 1 ? series : [];
  const hovered = hover !== null && hover < points.length ? points[hover] : null;
  const tooltipLeft = hover !== null ? x(hover) : 0;
  const flip = tooltipLeft > MARGIN.left + innerWidth * 0.6;
  const barWidth = Math.max(Math.min(24, slot - 2), 1);
  const category = hovered && colorBy
    ? colorBy.categories.find((item) => item.value === hovered[colorBy.key]) : undefined;
  const tableRows = points.slice(-TABLE_ROWS).reverse();

  return (
    <figure className="chart">
      {legend.length > 0 && (
        <ul className="chart-legend">
          {legend.map((item) => (
            <li key={item.key}>
              <span className={kind === "bar" ? "key key-box" : "key"} style={{ background: item.color }} />
              {item.label}
            </li>
          ))}
        </ul>
      )}
      <div
        className="chart-plot"
        ref={ref}
        tabIndex={0}
        role="group"
        aria-label={`${label}. Use the left and right arrow keys to read values.`}
        onKeyDown={onKey}
        onFocus={() => setHover((current) => current ?? points.length - 1)}
        onBlur={() => setHover(null)}
        style={{ height }}
      >
        {width > 0 && points.length > 0 && (
          <svg
            width={width}
            height={height}
            role="img"
            aria-label={label}
            onPointerMove={onMove}
            onPointerDown={onMove}
            onPointerLeave={() => setHover(null)}
          >
            <defs>
              <clipPath id={clipId}>
                <rect x={MARGIN.left - 2} y={MARGIN.top - 4} width={innerWidth + 8} height={innerHeight + 8} />
              </clipPath>
            </defs>
            {yTicks.map((tick) => (
              <g key={tick}>
                <line className="grid" x1={MARGIN.left} x2={MARGIN.left + innerWidth} y1={y(tick)} y2={y(tick)} />
                <text className="tick" x={MARGIN.left - 8} y={y(tick)} dy="0.32em" textAnchor="end">
                  {formatY(tick)}
                </text>
              </g>
            ))}
            <line
              className="axis"
              x1={MARGIN.left}
              x2={MARGIN.left + innerWidth}
              y1={MARGIN.top + innerHeight}
              y2={MARGIN.top + innerHeight}
            />
            {xTicks.map((index) => (
              <text key={index} className="tick" x={x(index)} y={height - 8} textAnchor="middle">
                {tickLabel(index)}
              </text>
            ))}
            <g clipPath={`url(#${clipId})`}>
              {kind === "bar"
                ? points.map((point, index) => {
                    const v = value(point, series[0].key);
                    if (v === null || v <= 0) return null;
                    const top = y(v);
                    const base = y(Math.max(min, 0));
                    const radius = Math.min(4, barWidth / 2, Math.max(base - top, 0));
                    const left = x(index) - barWidth / 2;
                    const right = left + barWidth;
                    const d = `M${left},${base}V${top + radius}Q${left},${top} ${left + radius},${top}`
                      + `H${right - radius}Q${right},${top} ${right},${top + radius}V${base}Z`;
                    return (
                      <path key={point.t} d={d} fill={series[0].color} opacity={hover === null || hover === index ? 1 : 0.55} />
                    );
                  })
                : categoryPaths
                  ? categoryPaths.map((d, index) => (
                      <path key={colorBy!.categories[index].value} className="line" d={d} stroke={colorBy!.categories[index].color} />
                    ))
                  : paths.map((d, index) => (
                      <path key={series[index].key} className="line" d={d} stroke={series[index].color} />
                    ))}
            </g>
            {kind !== "bar" && !colorBy && series.map((item) => {
              const index = lastIndex(item.key);
              if (index < 0) return null;
              const v = value(points[index], item.key)!;
              if (log && v <= 0) return null;
              return <circle key={item.key} className="dot" cx={x(index)} cy={y(v)} r={4} fill={item.color} />;
            })}
            {hovered && kind !== "bar" && (
              <>
                <line className="crosshair" x1={x(hover!)} x2={x(hover!)} y1={MARGIN.top} y2={MARGIN.top + innerHeight} />
                {series.map((item) => {
                  const v = value(hovered, item.key);
                  if (v === null || (log && v <= 0)) return null;
                  return (
                    <circle
                      key={item.key}
                      className="dot"
                      cx={x(hover!)}
                      cy={y(v)}
                      r={4.5}
                      fill={category ? category.color : item.color}
                    />
                  );
                })}
              </>
            )}
          </svg>
        )}
        {width > 0 && points.length === 0 && <p className="chart-empty">No data in this period.</p>}
        {hovered && (
          <div
            className="chart-tooltip"
            role="status"
            style={flip ? { right: width - tooltipLeft + 12 } : { left: tooltipLeft + 12 }}
          >
            <div className="tooltip-title">{times ? dateTime(hovered.t) : date(hovered.t)}</div>
            {series.map((item) => {
              const v = value(hovered, item.key);
              return (
                <div className="tooltip-row" key={item.key}>
                  <span className="key" style={{ background: category ? category.color : item.color }} />
                  <strong>{v === null ? "—" : show(v)}</strong>
                  <span>{item.label}</span>
                </div>
              );
            })}
            {category && <div className="tooltip-note">Trend: {category.label}</div>}
          </div>
        )}
      </div>
      <details className="chart-table">
        <summary>Show the numbers as a table</summary>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">{times ? "Time" : "Date"}</th>
                {series.map((item) => <th scope="col" key={item.key}>{item.label}</th>)}
                {colorBy && <th scope="col">Trend</th>}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((point) => (
                <tr key={point.t}>
                  <th scope="row">{times ? dateTime(point.t) : date(point.t)}</th>
                  {series.map((item) => {
                    const v = value(point, item.key);
                    return <td key={item.key}>{v === null ? "—" : show(v)}</td>;
                  })}
                  {colorBy && (
                    <td>{colorBy.categories.find((item) => item.value === point[colorBy.key])?.label ?? "—"}</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {points.length > TABLE_ROWS && (
          <p className="muted small">Showing the most recent {TABLE_ROWS} of {points.length} rows.</p>
        )}
      </details>
    </figure>
  );
}
