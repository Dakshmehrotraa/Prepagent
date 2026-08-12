import PatternBadge from "./PatternBadge";

type ToolCall = {
  tool: string;
  input: Record<string, unknown>;
  output: any;
};

export default function AgentTrace({ trace }: { trace: ToolCall[] }) {
  if (!trace || trace.length === 0) return null;

  return (
    <div className="mt-3 border-l border-ink-600 pl-4">
      <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-mist/50">
        agent trace
      </div>
      <ol className="space-y-3">
        {trace.map((call, i) => (
          <li key={i} className="font-mono text-xs text-mist/80">
            <div className="flex items-baseline gap-2">
              <span className="text-teal-400">{String(i + 1).padStart(2, "0")}</span>
              <span className="text-paper">{call.tool}</span>
              <span className="text-mist/40">
                ({Object.entries(call.input).map(([k, v]) => `${k}="${v}"`).join(", ")})
              </span>
            </div>

            {call.tool === "classify_pattern" && call.output?.pattern && (
              <div className="mt-1 flex items-center gap-2 pl-6">
                <PatternBadge pattern={call.output.pattern} />
                <span className="text-mist/50">
                  {(call.output.confidence * 100).toFixed(0)}% · {call.output.source}
                </span>
              </div>
            )}

            {call.tool === "retrieve_similar_problems" && call.output?.results && (
              <ul className="mt-1 space-y-0.5 pl-6 text-mist/60">
                {call.output.results.map((r: any) => (
                  <li key={r.id}>
                    ↳ {r.title}{" "}
                    <span className="text-mist/30">
                      (sim {r.similarity?.toFixed(2)})
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
