import Chat from "@/components/Chat";

export default function Home() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-10">
      <header className="mb-8">
        <div className="mb-1 font-mono text-[11px] uppercase tracking-widest text-teal-400">
          agentic rag · dsa &amp; sql
        </div>
        <h1 className="font-display text-3xl text-paper">PrepAgent</h1>
        <p className="mt-2 max-w-xl text-sm leading-relaxed text-mist/60">
          Retrieves similar problems, classifies the underlying pattern with a
          LoRA-tuned classifier, and walks through an approach — instead of
          you manually searching for "what pattern is this."
        </p>
      </header>
      <Chat />
    </main>
  );
}
