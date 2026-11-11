import { useEffect, useRef, useState } from "react";
import type { Answer, ChatMessage, Risk } from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    /* not JSON */
  }
  return `The server answered with an error (${response.status}).`;
}

export async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, { signal, headers: { Accept: "application/json" } });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new ApiError(0, "Could not reach the server. Check that it is running and try again.");
  }
  if (!response.ok) throw new ApiError(response.status, await readError(response));
  return (await response.json()) as T;
}

export interface Loaded<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
}

/** Load a URL; while a new URL loads, the previous data stays so the page does not jump. */
export function useApi<T>(url: string | null): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({ data: null, error: null, loading: url !== null });
  const group = useRef<string | null>(null);

  useEffect(() => {
    if (url === null) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    // Data from another view (different path) must not linger; the same view with new filters may.
    const path = url.split("?")[0];
    const sameView = group.current === path;
    group.current = path;
    setState((previous) => ({ data: sameView ? previous.data : null, error: null, loading: true }));
    const controller = new AbortController();
    getJson<T>(url, controller.signal)
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((error) => {
        if (error.name === "AbortError") return;
        setState({ data: null, error: error as ApiError, loading: false });
      });
    return () => controller.abort();
  }, [url]);

  return state;
}

export function riskUrl(price: number, typicalMove: number, account: number, riskPct: number): string {
  const query = new URLSearchParams({
    price: String(price),
    typical_move: String(typicalMove),
    account: String(account),
    risk_pct: String(riskPct),
  });
  return `/api/risk?${query}`;
}

export type { Risk };

export async function askAssistant(slug: string, question: string, history: ChatMessage[]): Promise<Answer> {
  let response: Response;
  try {
    response = await fetch(`/api/assets/${slug}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        question,
        history: history.map(({ role, content }) => ({ role, content: content.slice(0, 4000) })),
      }),
    });
  } catch {
    throw new ApiError(0, "Could not reach the server. Check that it is running and try again.");
  }
  if (!response.ok) throw new ApiError(response.status, await readError(response));
  return (await response.json()) as Answer;
}
