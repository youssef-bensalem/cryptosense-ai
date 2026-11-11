// Shapes of the JSON the back end returns (see src/cryptosense/service.py).

export type RangeName = "1m" | "3m" | "1y" | "all";
export type PeriodName = "full" | "heldout";

export interface Meta {
  disclaimer: string;
  ranges: RangeName[];
  default_range: RangeName;
  held_out_start: string;
  max_question_chars: number;
  assistant_model_available: boolean;
}

export interface Asset {
  symbol: string;
  slug: string;
  name: string;
  has_data: boolean;
  data_date?: string;
  data_age_days?: number;
  price?: number | null;
  change_1d?: number | null;
  change_7d?: number | null;
  change_30d?: number | null;
}

export interface Insight {
  title: string;
  text: string;
  kind: "risk" | "context" | "warning";
  evidence: string;
}

export interface Today {
  symbol: string;
  name: string;
  data_date: string;
  data_age_days: number;
  tomorrow: string;
  next_weekday: string;
  price: number | null;
  change_1d: number | null;
  change_7d: number | null;
  change_30d: number | null;
  low_30d: number | null;
  high_30d: number | null;
  trend: string | null;
  trend_words: string;
  vol_regime: string | null;
  vol_words: string;
  method: string;
  swing_1d: number | null;
  swing_word: string;
  swing_name: string;
  range_low: number | null;
  range_high: number | null;
  forecast_track_record: string;
  insights: Insight[];
  intraday: { t: string; close: number | null }[] | null;
  intraday_supported: boolean;
  disclaimer: string;
}

export interface WeekPattern {
  name: string;
  description: string;
  present: boolean;
  evidence: string | null;
  meaning: string | null;
}

export interface Week {
  symbol: string;
  data_date: string;
  price: number;
  trend: string | null;
  trend_words: string;
  vol_regime: string | null;
  vol_words: string;
  change_20d: number | null;
  above_sma_100: boolean | null;
  horizon_days: number;
  method: string;
  swing_word: string;
  swing_name: string;
  move: number | null;
  range_low: number | null;
  range_high: number | null;
  patterns: WeekPattern[];
  replication_note: string | null;
  disclaimer: string;
}

// A type alias (not an interface) so the points can be handed to the generic chart.
export type PricePoint = {
  t: string;
  close: number | null;
  volume: number | null;
  trend: string | null;
};

export interface Prices {
  symbol: string;
  range: RangeName;
  last_close: number;
  change_1d: number | null;
  change_30d: number | null;
  points: PricePoint[];
}

export interface Conditions {
  symbol: string;
  data_date: string;
  trend: string | null;
  vol_regime: string | null;
  momentum_20d: number | null;
  distance_sma_100: number | null;
  volatility_20d_annual: number | null;
  efficiency_ratio_20d: number | null;
  trend_shares: Record<string, number>;
  caption: string;
}

export interface SetupCondition {
  feature: string;
  rule: string;
  value: number | null;
  met: boolean;
}

export interface SetupDetail {
  name: string;
  description: string;
  flagged: boolean;
  date: string;
  trend: string | null;
  conditions: SetupCondition[];
  past_episodes: number;
  past_mean_return: number | null;
  past_mean_se: number | null;
  base_same_trend_mean: number | null;
  gap: number | null;
  gap_in_standard_errors: number | null;
  recent_starts: string[];
}

export interface Patterns {
  symbol: string;
  typical_move_5d: number | null;
  swing_word: string;
  method: string;
  setups: SetupDetail[];
  caption: string;
  replication_note: string | null;
}

export interface Volatility {
  symbol: string;
  data_date: string;
  horizon_days: number;
  method: string;
  daily: number | null;
  typical_move: number | null;
  annualised: number | null;
  swing_word: string;
  caption: string;
  realized_name: string;
  range: RangeName;
  points: { t: string; forecast: number | null; realized: number | null }[];
  note: string;
}

export interface StrategyRow {
  name: string;
  total_return: number | null;
  annual_return: number | null;
  annual_volatility: number | null;
  return_per_risk: number | null;
  max_drawdown: number | null;
  trades: number | null;
  time_in_market: number | null;
  win_rate: number | null;
  avg_trade: number | null;
}

export interface Strategies {
  symbol: string;
  period: PeriodName;
  range: RangeName;
  strategies: string[];
  summary: StrategyRow[];
  points: ({ t: string } & Record<string, number | string | null>)[];
  caption: string;
  findings: string | null;
  replication_note: string | null;
}

export interface Risk {
  stop_price: number;
  stop_fraction: number;
  invest: number;
  loss_at_stop: number;
  exceeds_account: boolean;
}

export type AnswerSource = "fixed" | "model" | "withheld" | "unavailable";

export interface Answer {
  answer: string;
  source: AnswerSource;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  source?: AnswerSource;
}
