// Number and date formatting used across the interface.

const DASH = "—";

export function percent(value: number | null | undefined, digits = 1, signed = false): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  const text = (Math.abs(value) * 100).toFixed(digits) + "%";
  if (value < 0) return "−" + text;
  return signed ? "+" + text : text;
}

export function plusMinus(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return "±" + (Math.abs(value) * 100).toFixed(digits) + "%";
}

export function price(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  const digits = Math.abs(value) >= 1000 ? 0 : Math.abs(value) >= 1 ? 2 : 4;
  return value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function money(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function number(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function compact(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e9) return trim(value / 1e9) + "B";
  if (abs >= 1e6) return trim(value / 1e6) + "M";
  if (abs >= 1e4) return trim(value / 1e3) + "K";
  if (abs >= 100) return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (abs >= 1) return trim(value);
  return String(Number(value.toPrecision(2)));
}

/** Round to a few significant decimals and drop trailing zeros after the decimal point only. */
function trim(value: number): string {
  const digits = Math.abs(value) >= 100 ? 0 : Math.abs(value) >= 10 ? 1 : 2;
  return String(Number(value.toFixed(digits)));
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "2026-11-10" -> "10 Nov 2026"; times keep hours and minutes in UTC. */
export function date(text: string | null | undefined, withYear = true): string {
  if (!text) return DASH;
  const stamp = new Date(text.length === 10 ? text + "T00:00:00Z" : text);
  if (Number.isNaN(stamp.getTime())) return text;
  const day = `${stamp.getUTCDate()} ${MONTHS[stamp.getUTCMonth()]}`;
  return withYear ? `${day} ${stamp.getUTCFullYear()}` : day;
}

export function dateTime(text: string): string {
  const stamp = new Date(text);
  if (Number.isNaN(stamp.getTime())) return text;
  const hours = String(stamp.getUTCHours()).padStart(2, "0");
  const minutes = String(stamp.getUTCMinutes()).padStart(2, "0");
  return `${stamp.getUTCDate()} ${MONTHS[stamp.getUTCMonth()]}, ${hours}:${minutes} UTC`;
}

export function capitalize(text: string): string {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}
