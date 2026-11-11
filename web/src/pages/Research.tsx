import { useState } from "react";
import { useApi } from "../api";
import { Chart } from "../components/Chart";
import { Badge, Card, Delta, ErrorBox, Loading, Notice, RangeControl, Segmented, Stat } from "../components/ui";
import { capitalize, compact, date, number, percent, plusMinus, price } from "../format";
import type {
  Conditions, Patterns, PeriodName, Prices, RangeName, Strategies, StrategyRow, Volatility,
} from "../types";

export type ResearchSection = "overview" | "conditions" | "patterns" | "volatility" | "strategies";

export const RESEARCH_SECTIONS: { value: ResearchSection; label: string }[] = [
  { value: "overview", label: "Market overview" },
  { value: "conditions", label: "Market conditions" },
  { value: "patterns", label: "Patterns" },
  { value: "volatility", label: "Volatility" },
  { value: "strategies", label: "Strategy test results" },
];

const TREND_COLORS = {
  key: "trend",
  categories: [
    { value: "up", label: "Up", color: "var(--series-1)" },
    { value: "down", label: "Down", color: "var(--series-2)" },
    { value: "range", label: "Range", color: "var(--series-3)" },
  ],
};

const SERIES_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
const wholePercent = (value: number) => percent(value, 0);

interface SectionProps {
  slug: string;
  range: RangeName;
}

function Overview({ slug, range }: SectionProps) {
  const { data, error, loading } = useApi<Prices>(`/api/assets/${slug}/prices?range=${range}`);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading />;
  return (
    <div className={`stack ${loading ? "refreshing" : ""}`}>
      <div className="stat-row">
        <Stat label="Last close (USDT)" value={price(data.last_close)} />
        <Stat label="1-day change" value={<Delta value={data.change_1d} digits={2} />} />
        <Stat label="30-day change" value={<Delta value={data.change_30d} digits={2} />} />
      </div>
      <Card title={`${data.symbol} daily close (log scale)`}>
        <Chart
          label={`${data.symbol} daily close in USDT on a log scale`}
          points={data.points}
          series={[{ key: "close", label: "Close (USDT)", color: "var(--series-1)" }]}
          log
          formatY={price}
        />
      </Card>
      <Card title={`${data.symbol} daily volume (base asset units)`}>
        <Chart
          label={`${data.symbol} daily volume in base asset units`}
          points={data.points}
          series={[{ key: "volume", label: "Volume", color: "var(--series-1)" }]}
          kind="bar"
          height={200}
          formatY={compact}
          formatValue={(value) => value.toLocaleString("en-US", { maximumFractionDigits: 0 })}
        />
      </Card>
    </div>
  );
}

function ConditionsSection({ slug, range }: SectionProps) {
  const conditions = useApi<Conditions>(`/api/assets/${slug}/conditions`);
  const prices = useApi<Prices>(`/api/assets/${slug}/prices?range=${range}`);
  if (conditions.error) return <ErrorBox error={conditions.error} />;
  if (!conditions.data) return <Loading />;
  const data = conditions.data;
  return (
    <div className="stack">
      <h3 className="section-title">Current conditions ({date(data.data_date)})</h3>
      <div className="stat-row">
        <Stat label="Trend regime" value={data.trend ?? "—"} />
        <Stat label="Volatility regime" value={data.vol_regime ?? "—"} />
      </div>
      <p className="muted small">{data.caption}</p>
      <div className="stat-row">
        <Stat label="20-day momentum" value={percent(data.momentum_20d, 1, true)} />
        <Stat label="Distance from 100-day average" value={percent(data.distance_sma_100, 1, true)} />
        <Stat label="Volatility (20d, annualised)" value={percent(data.volatility_20d_annual, 0)} />
        <Stat label="Efficiency ratio (20d)" value={number(data.efficiency_ratio_20d)} />
      </div>
      <Card title={`${data.symbol} trend regimes`}>
        {prices.error ? <ErrorBox error={prices.error} /> : !prices.data ? <Loading /> : (
          <div className={prices.loading ? "refreshing" : ""}>
            <Chart
              label={`${data.symbol} daily close on a log scale, colored by trend regime`}
              points={prices.data.points.filter((point) => point.trend !== null)}
              series={[{ key: "close", label: "Close (USDT)", color: "var(--series-1)" }]}
              colorBy={TREND_COLORS}
              log
              formatY={price}
            />
          </div>
        )}
      </Card>
      <Card title="Share of days in each trend regime (all history)">
        <div className="table-scroll">
          <table>
            <thead><tr><th scope="col">Trend</th><th scope="col">Share of days</th></tr></thead>
            <tbody>
              {Object.entries(data.trend_shares).map(([label, share]) => (
                <tr key={label}><th scope="row">{label}</th><td>{percent(share)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function PatternsSection({ slug }: { slug: string }) {
  const { data, error } = useApi<Patterns>(`/api/assets/${slug}/patterns`);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading />;
  return (
    <div className="stack">
      <p className="muted">{data.caption}</p>
      {data.replication_note && <Notice kind="warning">{data.replication_note}</Notice>}
      {data.setups.map((setup) => (
        <Card key={setup.name}>
          <div className="pattern-head">
            <h3 className="card-title">{capitalize(setup.name)} setup</h3>
            <Badge tone={setup.flagged ? "accent" : "neutral"}>
              {setup.flagged ? "Flagged" : "Not flagged"} on {date(setup.date)}
            </Badge>
          </div>
          <p className="muted">{capitalize(setup.description)}.</p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr><th scope="col">Feature</th><th scope="col">Rule</th><th scope="col">Value</th><th scope="col">Met</th></tr>
              </thead>
              <tbody>
                {setup.conditions.map((condition) => (
                  <tr key={condition.feature}>
                    <th scope="row"><code>{condition.feature}</code></th>
                    <td><code>{condition.rule}</code></td>
                    <td>{condition.value === null ? "—" : number(condition.value, 4)}</td>
                    <td>{condition.met ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="stat-row">
            <Stat
              label={`Typical 5-day ${data.swing_word}`}
              value={plusMinus(data.typical_move_5d)}
              hint={`Method: ${data.method}`}
            />
            <Stat label="Past episodes" value={setup.past_episodes} />
            {setup.past_mean_return !== null && (
              <>
                <Stat
                  label="Past avg 5-day return"
                  value={percent(setup.past_mean_return, 2, true)}
                  hint={`Standard error ±${percent(setup.past_mean_se, 2)}`}
                />
                <Stat label="Avg day in same trend regime" value={percent(setup.base_same_trend_mean, 2, true)} />
              </>
            )}
          </div>
          {setup.gap !== null && setup.gap_in_standard_errors !== null && (
            <p className="muted small">
              Difference from an average {setup.trend}-trend day: {percent(setup.gap, 2, true)} (
              {number(setup.gap_in_standard_errors, 1)} standard errors). Below about 2 standard errors, the
              difference could be chance.
            </p>
          )}
          <p className="small">
            Most recent episode starts: {setup.recent_starts.length ? setup.recent_starts.map((day) => date(day)).join(", ") : "none"}
          </p>
        </Card>
      ))}
    </div>
  );
}

function VolatilitySection({ slug, range }: SectionProps) {
  const { data, error, loading } = useApi<Volatility>(`/api/assets/${slug}/volatility?range=${range}`);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading />;
  return (
    <div className={`stack ${loading ? "refreshing" : ""}`}>
      <h3 className="section-title">
        Volatility forecast for the next {data.horizon_days} days ({date(data.data_date)})
      </h3>
      <p>Method: <strong>{data.method}</strong></p>
      <div className="stat-row">
        <Stat label="Daily volatility" value={percent(data.daily, 2)} />
        <Stat label={`Typical ${data.horizon_days}-day ${data.swing_word}`} value={plusMinus(data.typical_move)} />
        <Stat label="Annualised" value={percent(data.annualised, 0)} />
      </div>
      <p className="muted small">{data.caption}</p>
      <Card title="Forecast vs realized volatility (annualised)">
        <Chart
          label="Forecast and realized volatility, annualised"
          points={data.points}
          series={[
            { key: "forecast", label: "Forecast", color: "var(--series-1)" },
            { key: "realized", label: capitalize(data.realized_name), color: "var(--series-2)" },
          ]}
          formatY={wholePercent}
          formatValue={(value) => percent(value, 1)}
        />
        <p className="muted small">{data.note}</p>
      </Card>
    </div>
  );
}

const STRATEGY_COLUMNS: { key: keyof StrategyRow; label: string; format: (value: number | null) => string }[] = [
  { key: "total_return", label: "Total return", format: (v) => percent(v) },
  { key: "annual_return", label: "Annual return", format: (v) => percent(v) },
  { key: "annual_volatility", label: "Annual volatility", format: (v) => percent(v) },
  { key: "return_per_risk", label: "Return per risk", format: (v) => number(v) },
  { key: "max_drawdown", label: "Max drawdown", format: (v) => percent(v) },
  { key: "trades", label: "Trades", format: (v) => number(v, 0) },
  { key: "time_in_market", label: "Time in market", format: (v) => percent(v) },
  { key: "win_rate", label: "Win rate", format: (v) => percent(v) },
  { key: "avg_trade", label: "Avg trade", format: (v) => percent(v) },
];

function StrategiesSection({ slug, range }: SectionProps) {
  const [period, setPeriod] = useState<PeriodName>("full");
  const { data, error, loading } = useApi<Strategies>(`/api/assets/${slug}/strategies?period=${period}&range=${range}`);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading />;
  return (
    <div className={`stack ${loading ? "refreshing" : ""}`}>
      <p className="muted">{data.caption}</p>
      {data.replication_note && <Notice kind="warning">{data.replication_note}</Notice>}
      <Segmented
        label="Period"
        options={[{ value: "full", label: "Full history" }, { value: "heldout", label: "Held-out test (2024 onward)" }]}
        value={period}
        onChange={setPeriod}
      />
      {data.summary.length === 0 ? (
        <p className="muted">Not enough history to test the strategy in this period.</p>
      ) : (
        <>
          <Card title="Summary">
            <div className="table-scroll">
              <table className="numeric">
                <thead>
                  <tr>
                    <th scope="col">Strategy</th>
                    {STRATEGY_COLUMNS.map((column) => <th scope="col" key={column.key}>{column.label}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {data.summary.map((row) => (
                    <tr key={row.name}>
                      <th scope="row">{row.name}</th>
                      {STRATEGY_COLUMNS.map((column) => (
                        <td key={column.key}>{column.format(row[column.key] as number | null)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="muted small">Return per risk = annual return divided by annual volatility.</p>
          </Card>
          <Card title="Account value (1.0 at the start of the backtest, log scale)">
            <Chart
              label="Account value of each strategy on a log scale, starting at 1.0"
              points={data.points}
              series={data.strategies.map((name, index) => ({ key: name, label: capitalize(name), color: SERIES_COLORS[index % 3] }))}
              log
              formatY={compact}
              formatValue={(value) => number(value, 3)}
            />
          </Card>
        </>
      )}
      {data.findings && <p className="muted small">{data.findings}</p>}
    </div>
  );
}

/** On narrow screens the section buttons scroll sideways; keep the selected one in view. */
function keepVisible(element: HTMLButtonElement | null) {
  element?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

export function Research({ slug, section, onSection, range, onRange }: {
  slug: string;
  section: ResearchSection;
  onSection: (section: ResearchSection) => void;
  range: RangeName;
  onRange: (range: RangeName) => void;
}) {
  const usesRange = section !== "patterns";
  return (
    <div className="stack">
      <header className="page-head">
        <h2>Research details</h2>
        <p className="muted">The technical numbers behind the summaries. Charts show daily data.</p>
      </header>
      <div className="filter-row">
        <nav className="subnav" aria-label="Research sections">
          {RESEARCH_SECTIONS.map((item) => (
            <button
              key={item.value}
              type="button"
              className={item.value === section ? "active" : ""}
              aria-current={item.value === section ? "page" : undefined}
              ref={item.value === section ? keepVisible : undefined}
              onClick={() => onSection(item.value)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        {usesRange && <RangeControl value={range} onChange={onRange} />}
      </div>
      {section === "overview" && <Overview slug={slug} range={range} />}
      {section === "conditions" && <ConditionsSection slug={slug} range={range} />}
      {section === "patterns" && <PatternsSection slug={slug} />}
      {section === "volatility" && <VolatilitySection slug={slug} range={range} />}
      {section === "strategies" && <StrategiesSection slug={slug} range={range} />}
    </div>
  );
}
