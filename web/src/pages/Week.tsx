import { useApi } from "../api";
import { RiskHelper } from "../components/RiskHelper";
import { Badge, Card, ErrorBox, Loading, Notice } from "../components/ui";
import { capitalize, date, percent, plusMinus, price } from "../format";
import type { Week as WeekData } from "../types";

export function Week({ slug }: { slug: string }) {
  const { data, error } = useApi<WeekData>(`/api/assets/${slug}/week`);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading />;

  const direction = data.change_20d !== null && data.change_20d < 0 ? "fallen" : "risen";
  const position = data.above_sma_100 === null
    ? "there is not enough history yet to compare it with its 100-day average"
    : data.above_sma_100 ? "it is above its 100-day average" : "it is below its 100-day average";

  return (
    <div className="stack">
      <header className="page-head">
        <h2>{data.symbol}: {data.trend_words}, {data.vol_words}</h2>
        <p className="muted">Based on the last complete day, {date(data.data_date)}.</p>
      </header>

      <Card>
        <p className="lead">
          Over the past 20 days the price has {direction} {percent(data.change_20d === null ? null : Math.abs(data.change_20d))}, and {position}.
        </p>
      </Card>

      <Card title="How much could the price move this week?">
        {data.move === null ? (
          <p className="muted">Not enough history to estimate a typical move.</p>
        ) : (
          <>
            <div className="hero-figure small-hero">{plusMinus(data.move)}</div>
            <p>
              A {data.swing_name} over the next {data.horizon_days} days is about {plusMinus(data.move)}: from today's
              close of {price(data.price)}, that is roughly {price(data.range_low)} to {price(data.range_high)} USDT.
            </p>
            <p className="muted small">This is a typical {data.swing_word}; larger moves do happen. Method: {data.method}.</p>
          </>
        )}
      </Card>

      <Card title="Patterns">
        {data.replication_note && <Notice kind="warning">{data.replication_note}</Notice>}
        <ul className="pattern-list">
          {data.patterns.map((pattern) => (
            <li key={pattern.name}>
              <div className="pattern-head">
                <Badge tone={pattern.present ? "accent" : "neutral"}>{pattern.present ? "Pattern found" : "Not present today"}</Badge>
                <span>{capitalize(pattern.description)}</span>
              </div>
              {pattern.present && (
                <p>Evidence: <strong>{pattern.evidence}</strong>. {pattern.meaning}</p>
              )}
            </li>
          ))}
        </ul>
      </Card>

      <RiskHelper
        price={data.price}
        typicalMove={data.move}
        basis={`the typical ${data.horizon_days}-day ${data.swing_word}`}
        holding={`for about ${data.horizon_days} days`}
      />

      <p className="muted">{data.disclaimer} The Research tab shows the full technical details behind these numbers.</p>
    </div>
  );
}
