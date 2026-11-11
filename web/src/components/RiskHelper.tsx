import { useEffect, useState } from "react";
import { riskUrl, useApi } from "../api";
import { money, percent, price as formatPrice } from "../format";
import type { Risk } from "../types";
import { Card, Notice, Stat } from "./ui";

interface RiskHelperProps {
  price: number | null;
  typicalMove: number | null;
  /** Names the move the stop is based on, for example "tomorrow's typical swing". */
  basis: string;
  holding: string;
}

function parse(text: string): number | null {
  const value = Number(text);
  return text.trim() !== "" && Number.isFinite(value) ? value : null;
}

/** Stop and position size; the arithmetic is done by the back end (explain.position_size). */
export function RiskHelper({ price, typicalMove, basis, holding }: RiskHelperProps) {
  const [account, setAccount] = useState("1000");
  const [riskPct, setRiskPct] = useState("1");
  const [query, setQuery] = useState<string | null>(null);

  const accountValue = parse(account);
  const riskValue = parse(riskPct);
  const valid = price !== null && typicalMove !== null && typicalMove > 0
    && accountValue !== null && accountValue > 0 && riskValue !== null && riskValue > 0 && riskValue <= 100;

  useEffect(() => {
    if (!valid) {
      setQuery(null);
      return;
    }
    const timer = window.setTimeout(() => setQuery(riskUrl(price!, typicalMove!, accountValue!, riskValue!)), 250);
    return () => window.clearTimeout(timer);
  }, [valid, price, typicalMove, accountValue, riskValue]);

  const { data, error, loading } = useApi<Risk>(query);

  return (
    <Card title="Risk helper">
      <p className="muted">For a trade held {holding}. Choose how much you are prepared to lose; the helper sizes the position to match.</p>
      <div className="form-row">
        <label>
          <span>Account size (USDT)</span>
          <input type="number" inputMode="decimal" min="1" step="100" value={account} onChange={(event) => setAccount(event.target.value)} />
        </label>
        <label>
          <span>Share of the account you are willing to lose on one trade (%)</span>
          <input type="number" inputMode="decimal" min="0.1" max="100" step="0.5" value={riskPct} onChange={(event) => setRiskPct(event.target.value)} />
        </label>
      </div>
      {price === null || typicalMove === null ? (
        <p className="muted">Not enough history to suggest a stop.</p>
      ) : !valid ? (
        <p className="muted">Enter an account size above 0 and a risk between 0 and 100%.</p>
      ) : error ? (
        <p className="muted">{error.message}</p>
      ) : !data ? (
        <p className="muted">Calculating…</p>
      ) : (
        <div className={loading ? "refreshing" : ""}>
          <div className="stat-row">
            <Stat
              label="Suggested stop"
              value={`${formatPrice(data.stop_price)} USDT`}
              hint={`${percent(data.stop_fraction)} below today's close`}
            />
            <Stat label="Position size" value={`${money(data.invest)} USDT`} />
          </div>
          <p>
            Placing the stop at twice {basis}, investing {money(data.invest)} USDT means that hitting the stop
            loses about {money(data.loss_at_stop)} USDT ({riskValue}% of the account), before costs.
          </p>
          {data.exceeds_account && (
            <Notice kind="warning">
              This is more than the account size; it would require borrowing, which this tool does not assume.
            </Notice>
          )}
          <p className="muted small">Prices can jump past a stop in fast markets.</p>
        </div>
      )}
    </Card>
  );
}
