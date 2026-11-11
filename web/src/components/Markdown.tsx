import type { ReactNode } from "react";

// A small, safe renderer for the assistant's answers: paragraphs, bullet lists,
// **bold** and *italic*. Text is never inserted as HTML.

function inline(text: string, keyPrefix: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|\*[^*\n]+\*)/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let index = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    const token = match[0];
    parts.push(token.startsWith("**")
      ? <strong key={`${keyPrefix}-${index}`}>{token.slice(2, -2)}</strong>
      : <em key={`${keyPrefix}-${index}`}>{token.slice(1, -1)}</em>);
    last = match.index + token.length;
    index++;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

export function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let paragraph: string[] = [];
  let items: string[] = [];
  const flush = () => {
    if (paragraph.length) {
      blocks.push(<p key={blocks.length}>{inline(paragraph.join(" "), `p${blocks.length}`)}</p>);
      paragraph = [];
    }
    if (items.length) {
      const key = blocks.length;
      blocks.push(<ul key={key}>{items.map((item, i) => <li key={i}>{inline(item, `l${key}-${i}`)}</li>)}</ul>);
      items = [];
    }
  };
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const bullet = /^[-*•]\s+(.*)$/.exec(line);
    if (bullet) {
      if (paragraph.length) flush();
      items.push(bullet[1]);
    } else if (line === "") {
      flush();
    } else {
      if (items.length) flush();
      paragraph.push(line);
    }
  }
  flush();
  return <div className="markdown">{blocks}</div>;
}
