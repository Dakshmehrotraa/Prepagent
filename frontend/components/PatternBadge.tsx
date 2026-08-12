const PATTERN_COLORS: Record<string, string> = {
  two_pointers: "border-teal-400 text-teal-400",
  sliding_window: "border-amber text-amber",
  binary_search: "border-mist text-mist",
  dfs_backtracking: "border-teal-400 text-teal-400",
  bfs_graph: "border-amber text-amber",
  dynamic_programming: "border-mist text-mist",
  greedy: "border-teal-400 text-teal-400",
  heap_priority_queue: "border-amber text-amber",
  union_find: "border-mist text-mist",
  prefix_sum_hashing: "border-teal-400 text-teal-400",
};

export default function PatternBadge({ pattern }: { pattern: string }) {
  const cls = PATTERN_COLORS[pattern] ?? "border-mist text-mist";
  return (
    <span
      className={`inline-block rounded-sm border px-2 py-0.5 font-mono text-[11px] tracking-tight ${cls}`}
    >
      {pattern.replaceAll("_", " · ")}
    </span>
  );
}
