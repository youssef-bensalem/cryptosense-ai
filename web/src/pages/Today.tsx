import { useApi } from "../api";
import { Chart } from "../components/Chart";
import { RiskHelper } from "../components/RiskHelper";
import { Card, Delta, ErrorBox, Loading, Notice, Stat } from "../components/ui";
import { capitalize, date, plusMinus, price } from "../format";
import type { Today as TodayData } from "../types";

export function Today({ slug }: { slug: string }) {
  const { data, error } = useApi<TodayData>(`/api/assets/${slug}/today`);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading />;

  const warnings = data.insights.filter((insight) => insight.kind === "warning");
  const others = data.insights.filter((insight) => insight.kind !== "warning");

  return (
    <div className="stack">
      <header className="page-head">
        <h2>{data.symbol}: what to expect tomorrow</h2>
        <p className="muted">
          {data.next_weekday}, {date(data.tomorrow)}. Based on the last complete day, {date(data.data_date)}. Method: {data.method}.
        </p>
      </header>

      {warnings.map((insight) => (
        <Notice key={insight.title} kind="warning" title={insight.title}>
          {insight.text}
        </Notice>
      ))}

      <div className="grid-hero">
        <Card className="hero">
          {data.swing_1d === null ? (
            <p className="muted">Not enough history to estimate tomorrow's typical swing.</p>
          ) : (
            <>
              <div className="stat-label">Tomorrow's {data.swing_name}</div>
              <div className="hero-figure">{plusMinus(data.swing_1d)}</div>
              <div className="hero-range">
                <span className="stat-label">Typical range</span>
                <strong>{price(data.range_low)} – {price(data.range_high)} USDT</strong>
              </div>
              <p className="muted">
                From today's close of {price(data.price)} USDT. This is a typical size of move, not a direction;
                larger moves do happen.
              </p>
            </>
          )}
        </Card>
        <Card title="Where things stand">
          <div className="stat-grid">
            <Stat label="Last close" value={`${price(data.price)} USDT`} />
            <Stat label="1 day" value={<Delta value={data.change_1d} />} />
            <Stat label="7 days" value={<Delta value={data.change_7d} />} />
            <Stat label="30 days" value={<Delta value={data.change_30d} />} />
          </div>
          <dl className="facts">
            <div><dt>Trend</dt><dd>{capitalize(data.trend_words)}</dd></div>
            <div><dt>Turbulence</dt><dd>{capitalize(data.vol_words)}</dd></div>
            <div><dt>30-day range</dt><dd>{price(data.low_30d)} – {price(data.high_30d)} USDT</dd></div>
          </dl>
        </Card>
      </div>

      <section>
        <h3 className="section-title">What this means</h3>
        <div className="insights">
          {others.map((insight) => (
            <article key={insight.title} className={`insight insight-${insight.kind}`}>
              <h4>{insight.title}</h4>
              <p>{insight.text}</p>
              <p className="muted small">Evidence: {insight.evidence}</p>
            </article>
          ))}
        </div>
      </section>

      <RiskHelper
        price={data.price}
        typicalMove={data.swing_1d}
        basis={`tomorrow's typical ${data.swing_word}`}
        holding="for one day"
      />

      <Card title="Last 3 days, every 15 minutes">
        {data.intraday ? (
          <Chart
            label={`${data.symbol} 15-minute close over the last 3 days, in UTC`}
            points={data.intraday}
            series={[{ key: "close", label: "Close (USDT)", color: "var(--series-1)" }]}
            times
            formatY={price}
          />
        ) : (
          <p className="muted">
            {data.intraday_supported
              ? "No 15-minute data has been downloaded for this asset yet."
              : "No intraday data for this asset."}
          </p>
        )}
      </Card>

      <p className="muted">{data.disclaimer}</p>
    </div>
  );
}
