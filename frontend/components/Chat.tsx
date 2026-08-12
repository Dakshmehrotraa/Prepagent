"use client";

import { useState, useRef, useEffect } from "react";
import AgentTrace from "./AgentTrace";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type Message = {
  role: "user" | "agent";
  content: string;
  trace?: any[];
};

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send() {
    const text = input.trim();
    if (!text || loading) return;

    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/agent`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setMessages((m) => [
        ...m,
        { role: "agent", content: data.answer, trace: data.trace },
      ]);
    } catch (err: any) {
      setMessages((m) => [
        ...m,
        {
          role: "agent",
          content: `Couldn't reach PrepAgent's backend (${err.message}). Is the API running?`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-[calc(100vh-140px)] flex-col rounded border border-ink-700 bg-ink-900">
      <div className="flex-1 overflow-y-auto p-6">
        {messages.length === 0 && (
          <div className="font-mono text-sm text-mist/40">
            Paste a problem statement, or ask something like{" "}
            <span className="text-mist/70">
              "how do I approach finding the longest substring without repeats?"
            </span>
          </div>
        )}

        <div className="space-y-6">
          {messages.map((m, i) => (
            <div key={i}>
              <div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-mist/40">
                {m.role === "user" ? "you" : "prepagent"}
              </div>
              <div
                className={
                  m.role === "user"
                    ? "text-paper"
                    : "whitespace-pre-wrap leading-relaxed text-paper"
                }
              >
                {m.content}
              </div>
              {m.trace && <AgentTrace trace={m.trace} />}
            </div>
          ))}
          {loading && (
            <div className="font-mono text-xs text-teal-400/70">
              retrieving context, classifying pattern…
            </div>
          )}
        </div>
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-ink-700 p-4">
        <div className="flex items-end gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            rows={2}
            placeholder="Describe the problem you're stuck on…"
            className="flex-1 resize-none rounded border border-ink-600 bg-ink-800 p-3 text-sm text-paper placeholder:text-mist/30 focus:border-teal-400 focus:outline-none"
          />
          <button
            onClick={send}
            disabled={loading}
            className="rounded bg-teal-500 px-4 py-2.5 text-sm font-medium text-ink-950 transition hover:bg-teal-400 disabled:opacity-40"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
