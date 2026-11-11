import { useCallback, useEffect, useState } from "react";
import { useApi } from "./api";
import { Delta, ErrorBox, Loading, Notice } from "./components/ui";
import { price } from "./format";
import { Assistant } from "./pages/Assistant";
import { RESEARCH_SECTIONS, Research } from "./pages/Research";
import type { ResearchSection } from "./pages/Research";
import { Today } from "./pages/Today";
import { Week } from "./pages/Week";
import type { Asset, ChatMessage, Meta, RangeName } from "./types";

type Page = "today" | "week" | "assistant" | "research";

const PAGES: { value: Page; label: string }[] = [
  { value: "today", label: "Today & tomorrow" },
  { value: "week", label: "This week" },
  { value: "assistant", label: "Ask the assistant" },
  { value: "research", label: "Research" },
];

interface Route {
  slug: string | null;
  page: Page;
  section: ResearchSection;
}

function parseHash(hash: string): Route {
  const [slug, page, section] = hash.replace(/^#\/?/, "").split("/");
  return {
    slug: slug || null,
    page: PAGES.some((item) => item.value === page) ? (page as Page) : "today",
    section: RESEARCH_SECTIONS.some((item) => item.value === section) ? (section as ResearchSection) : "overview",
  };
}

function useRoute(): [Route, (next: Partial<Route>) => void] {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const go = useCallback((next: Partial<Route>) => {
    const target = { ...parseHash(window.location.hash), ...next };
    const parts = [target.slug ?? "", target.page];
    if (target.page === "research") parts.push(target.section);
    window.location.hash = "/" + parts.join("/");
  }, []);
  return [route, go];
}

type Theme = "light" | "dark";

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const saved = window.localStorage.getItem("theme");
      if (saved === "light" || saved === "dark") return saved;
    } catch {
      /* storage unavailable */
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    try {
      window.localStorage.setItem("theme", next);
    } catch {
      /* storage unavailable */
    }
  };
  return [theme, toggle];
}

/** On narrow screens the page tabs scroll sideways; keep the selected one in view. */
function keepVisible(element: HTMLButtonElement | null) {
  element?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

export function App() {
  const [route, go] = useRoute();
  const [theme, toggleTheme] = useTheme();
  const [range, setRange] = useState<RangeName>("3m");
  const [chats, setChats] = useState<Record<string, ChatMessage[]>>({});
  const assets = useApi<Asset[]>("/api/assets");
  const meta = useApi<Meta>("/api/meta");

  const available = assets.data?.filter((asset) => asset.has_data) ?? [];
  const current = assets.data?.find((asset) => asset.slug === route.slug) ?? available[0] ?? assets.data?.[0] ?? null;

  useEffect(() => {
    document.title = current ? `${current.symbol} · CryptoSense AI` : "CryptoSense AI";
  }, [current]);

  return (
    <div className="app">
      <a className="skip" href="#main">Skip to content</a>
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#/" aria-label="CryptoSense AI home">
            <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden="true">
              <rect width="32" height="32" rx="8" fill="var(--accent)" />
              <path d="M6 21l6-7 5 4 9-10" fill="none" stroke="#fff" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span>CryptoSense <span className="brand-ai">AI</span></span>
          </a>
          <nav className="assets" aria-label="Asset">
            {(assets.data ?? []).map((asset) => (
              <button
                key={asset.slug}
                type="button"
                className={`asset ${current?.slug === asset.slug ? "active" : ""}`}
                aria-pressed={current?.slug === asset.slug}
                onClick={() => go({ slug: asset.slug })}
              >
                <span className="asset-symbol">{asset.symbol.split("/")[0]}</span>
                {asset.has_data ? (
                  <>
                    <span className="asset-price">{price(asset.price)}</span>
                    <Delta value={asset.change_1d} />
                  </>
                ) : (
                  <span className="muted small">no data</span>
                )}
              </button>
            ))}
          </nav>
          <button type="button" className="icon-button" onClick={toggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </div>
        <nav className="tabs" aria-label="Pages">
          <div className="tabs-inner">
            {PAGES.map((item) => (
              <button
                key={item.value}
                type="button"
                className={route.page === item.value ? "active" : ""}
                aria-current={route.page === item.value ? "page" : undefined}
                ref={route.page === item.value ? keepVisible : undefined}
                onClick={() => go({ slug: current?.slug ?? null, page: item.value })}
              >
                {item.label}
              </button>
            ))}
          </div>
        </nav>
      </header>

      <main id="main" className="content">
        {assets.error ? (
          <ErrorBox error={assets.error} />
        ) : !assets.data ? (
          <Loading />
        ) : !current || !current.has_data ? (
          <Notice title="No data yet">
            No market data has been downloaded for {current ? current.symbol : "any asset"} yet. Run the data refresh
            on the computer that hosts this project, then reload this page.
          </Notice>
        ) : (
          <>
            {route.page === "today" && <Today slug={current.slug} />}
            {route.page === "week" && <Week slug={current.slug} />}
            {route.page === "assistant" && (
              <Assistant slug={current.slug} symbol={current.symbol} meta={meta.data} chats={chats} onChats={setChats} />
            )}
            {route.page === "research" && (
              <Research
                slug={current.slug}
                section={route.section}
                onSection={(section) => go({ slug: current.slug, page: "research", section })}
                range={range}
                onRange={setRange}
              />
            )}
          </>
        )}
      </main>

      <footer className="footer">
        <p>{meta.data?.disclaimer ?? "Research and decision support only. Not financial advice; no trades are placed."}</p>
        <p className="small">
          Prices are daily and 15-minute Binance candles in UTC, updated when the data is refreshed
          {current?.data_date ? `; latest complete day: ${current.data_date}` : ""}.
        </p>
      </footer>
    </div>
  );
}
