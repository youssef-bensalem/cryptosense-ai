import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { ApiError, askAssistant } from "../api";
import { Markdown } from "../components/Markdown";
import { Badge, Notice } from "../components/ui";
import type { AnswerSource, ChatMessage, Meta } from "../types";

const SUGGESTIONS = [
  "How much could the price move tomorrow?",
  "What does “typical swing” mean?",
  "Why do trading costs matter today?",
  "Should I buy now?",
];

const SOURCE_LABELS: Record<AnswerSource, { label: string; tone: "neutral" | "accent" | "warning" }> = {
  fixed: { label: "Checked answer (no AI model)", tone: "accent" },
  model: { label: "AI model, limited to the dashboard's facts", tone: "neutral" },
  withheld: { label: "AI answer withheld; checked answer shown", tone: "warning" },
  unavailable: { label: "AI model unavailable", tone: "warning" },
};

const HISTORY_SENT = 12;

export function Assistant({ slug, symbol, meta, chats, onChats }: {
  slug: string;
  symbol: string;
  meta: Meta | null;
  chats: Record<string, ChatMessage[]>;
  onChats: (chats: Record<string, ChatMessage[]>) => void;
}) {
  const history = chats[slug] ?? [];
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const end = useRef<HTMLDivElement | null>(null);
  const maxLength = meta?.max_question_chars ?? 500;

  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest" });
  }, [history.length, busy]);

  const send = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    const asked: ChatMessage[] = [...history, { role: "user", content: trimmed }];
    onChats({ ...chats, [slug]: asked });
    setQuestion("");
    setBusy(true);
    setError(null);
    try {
      const reply = await askAssistant(slug, trimmed, history.slice(-HISTORY_SENT));
      onChats({ ...chats, [slug]: [...asked, { role: "assistant", content: reply.answer, source: reply.source }] });
    } catch (problem) {
      setError(problem instanceof ApiError ? problem.message : "Something went wrong. Please try again.");
      onChats({ ...chats, [slug]: history });
      setQuestion(trimmed);
    } finally {
      setBusy(false);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void send(question);
  };

  return (
    <div className="stack">
      <header className="page-head">
        <h2>Ask about {symbol}</h2>
        <p className="muted">
          Answers use only the facts and deductions on the Today &amp; tomorrow tab. Questions about buying, selling
          or price direction always get a fixed, checked answer. Information, not financial advice.
        </p>
      </header>

      {meta && !meta.assistant_model_available && (
        <Notice>
          No AI model is connected right now, so only questions about buying, selling or price direction can be
          answered. Everything else in the dashboard still works.
        </Notice>
      )}

      <div className="chat" aria-live="polite">
        {history.length === 0 && (
          <div className="chat-empty">
            <p className="muted">Try one of these:</p>
            <div className="chips">
              {SUGGESTIONS.map((suggestion) => (
                <button key={suggestion} type="button" className="chip" onClick={() => void send(suggestion)} disabled={busy}>
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
        {history.map((message, index) => (
          <div key={index} className={`bubble bubble-${message.role}`}>
            {message.role === "assistant" ? (
              <>
                {message.source && (
                  <Badge tone={SOURCE_LABELS[message.source].tone}>{SOURCE_LABELS[message.source].label}</Badge>
                )}
                <Markdown text={message.content} />
              </>
            ) : (
              <p>{message.content}</p>
            )}
          </div>
        ))}
        {busy && <div className="bubble bubble-assistant"><p className="muted">Thinking…</p></div>}
        <div ref={end} />
      </div>

      {error && <Notice kind="warning">{error}</Notice>}

      <form className="chat-form" onSubmit={submit}>
        <label className="visually-hidden" htmlFor="question">Your question about {symbol}</label>
        <input
          id="question"
          type="text"
          value={question}
          maxLength={maxLength}
          placeholder={`Ask about ${symbol} today`}
          onChange={(event) => setQuestion(event.target.value)}
          autoComplete="off"
        />
        <button type="submit" className="primary" disabled={busy || question.trim() === ""}>Ask</button>
      </form>
      {history.length > 0 && (
        <button type="button" className="link" onClick={() => onChats({ ...chats, [slug]: [] })} disabled={busy}>
          Clear this conversation
        </button>
      )}
    </div>
  );
}
