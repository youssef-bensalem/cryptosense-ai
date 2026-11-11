# Pre-registration: news sentiment and next-day returns

**Status:** Frozen on 2026-10-28, by the commit "Freeze news pre-registration". Nothing in this file changes after that commit; any deviation is reported as a deviation, with its date and reason.

**Written before** any daily sentiment score has been computed or compared with prices. Only data-quality figures (article counts, collector and labeler run logs, label quality checks) have been looked at.

## 1. Question

Does the overall mood of crypto news on day D predict the return of BTC/USDT and SOL/USDT on the following day, beyond what day D's own price move already predicts?

- **H1 (primary):** daily market news sentiment on day D has a positive relationship with the next-day return, for BTC and for SOL, after controlling for day D's return.
- **H2 (secondary):** daily BTC-specific news sentiment on day D has a positive relationship with BTC's next-day return, after the same control.

No hypothesis is tested for SOL-specific sentiment: in the first 124 articles, 1 mentioned Solana.

## 2. Data (fixed)

- **Sources:** coindesk, cointelegraph, decrypt, theblock, bitcoinmagazine. No source is added or removed during the study.
- **Collection start:** `COLLECTION_START = 2026-10-24T22:01:53Z` (from `cryptosense.news`). Articles published before it are excluded.
- **Labels:** model `gpt-oss-120b` (reported `openai/gpt-oss-120b`), prompt version `affefb1b22ff`, frozen. Labels from any other model or prompt version are excluded.
- **Prices:** Binance daily candles for BTC/USDT and SOL/USDT from the project's data pipeline (UTC days).

## 3. Study period

- **First study day:** 2026-10-25 (the first complete UTC day after the collection start).
- **Last study day:** 2027-01-31. The evaluation is run once, after the outcome for the last study day is known (from 2027-02-03).
- The study is not stopped early and not extended, whatever interim data quality looks like.

## 4. Article rules

An article counts toward day D only if all of these hold:

1. `published_utc >= COLLECTION_START`.
2. Its UTC publication date is D (day D runs from 00:00 to 24:00 UTC; the signal is known at D's close).
3. It was fetched within 24 hours of publication (`fetched_utc - published_utc <= 24h`), to exclude backdated or re-published items.
4. It has a valid label from the frozen model and prompt.
5. It is not a multi-story newsletter: a decrypt article whose title contains two or more "!" characters. (Checked on 2026-10-28 against the collected titles only, without labels or prices: the rule matched 18 newsletter titles and no other title; no decrypt title contained exactly one "!".)
6. Exact duplicates are counted once: articles whose titles are identical after lowercasing and removing punctuation, published within 48 hours of each other, form one item whose sentiment is the mean of its members.

## 5. Daily scores

- **Market sentiment S_market(D):** mean of `sentiment_market` over day D's items with `relevance_market = true`.
- **BTC sentiment S_btc(D):** mean of `sentiment_btc` over day D's items with `relevance_btc = true`.
- Both scores are standardized (z-scores) using the mean and standard deviation over all study days, for reporting only; the test statistic does not depend on this scaling.

## 6. Coverage: which days count

Day D is included only if:

- the collector ran successfully in at least 18 of the 24 hours of day D (from `data/news/runs.csv`), and
- day D has at least 5 market-relevant items (for H1) or at least 3 BTC-relevant items (for H2).

Excluded days are listed with the reason. If fewer than 60 days qualify for H1, the result is reported as inconclusive for lack of data.

## 7. Outcome

Next-day return, as in the one-day track: signal at the close of day D, entry at the open of D+1, exit at the open of D+2:

`R(D) = ln(open[D+2] / open[D+1])`

Control: day D's own return, `ln(close[D] / close[D-1])`.

## 8. Test

For each asset (H1: BTC and SOL; H2: BTC), an ordinary least squares regression over the included days:

`R(D) = a + b * S(D) + c * r(D) + error`

with heteroskedasticity-robust (HC3) standard errors.

- **Statistic:** the t-statistic of b.
- **Success for H1, per asset:** b > 0 and t >= 2.24. (Two primary tests; 2.24 is the two-sided 5% threshold after a Bonferroni correction for two tests.)
- **Success for H2:** b > 0 and t >= 2.0 (secondary; reported, but not used for the dashboard decision).
- Also reported, not used for the decision: the simple correlation of S(D) and R(D), the coefficient without the control, the number of included days, and the result with b < 0 noted as "negative relationship" if t <= -2.24.

**Statistical power, stated in advance:** with about 90 to 100 days, only a strong relationship (a correlation of roughly 0.2 or more) can reach the threshold. A non-significant result therefore means "no strong effect was found", not "no effect exists".

## 9. Robustness checks (reported, never used to change the conclusion)

- Without the duplicate rule (every article counted).
- Without CoinDesk (headline-only articles).
- Weighting each item by the label's `confidence`.
- Using only days with at least 10 market-relevant items.

## 10. What happens next

- **H1 succeeds for an asset:** a "News mood" card is added to that asset's Summary view, showing today's sentiment and an evidence badge based on this result.
- **H1 does not succeed:** the result is recorded as negative. The dashboard may show news mood as context only, labeled as having no tested predictive value.
- Either way, the results, every excluded day, and any deviation from this plan are written up in the notebook and the roadmap.

## 11. Rules until the evaluation

- No daily sentiment score is placed next to any price or return before the evaluation date.
- Allowed meanwhile: data-quality monitoring only (article counts, collector and labeler failures, label coverage per day).
- The collector, labeler, model, prompt and sources do not change. If something breaks (a source stops working, the model is withdrawn), it is recorded as a deviation with its date, and the affected days are excluded rather than relabeled.

## 12. Known limitations

- Label quality: sentiment direction agreed with hand labels less often than intended (market 30%, BTC 67%; bar 75%), mainly because the model rates ordinary positive news +1 where hand labels used 0. Consistency between runs was 92-100%.
- Coverage depends on a home PC being on; gaps are measured, not filled.
- Headline-only articles (CoinDesk) carry less information.
- One model, one prompt, five sources, one period: a positive result would need replication before being relied on.
