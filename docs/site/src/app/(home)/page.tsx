'use client';

import { useState } from 'react';
import Link from 'next/link';

type Pillar = {
  key: string;
  tab: string;
  file: string;
  lines: string[];
  highlight: number;
  calloutLabel: string;
  calloutTitle: string;
  calloutBody: string;
};

const PILLARS: Pillar[] = [
  {
    key: 'agent',
    tab: 'Agent',
    file: 'agent.rs',
    lines: [
      'let mut agent = AgentBuilder::new()',
      '    .with_llm(Client::from_env()?)',
      '    .with_tool(Arc::new(Calculator::new()))',
      '    .with_max_iterations(4)',
      '    .build()?;',
    ],
    highlight: 2,
    calloutLabel: '01 · TOOLS',
    calloutTitle: 'Tool dispatch, checked at compile time.',
    calloutBody:
      'Calculator implements Tool. Swap it for your own impl and the schema the model sees is derived from your Rust type — it cannot drift from the code that runs.',
  },
  {
    key: 'graph',
    tab: 'Graph',
    file: 'graph.rs',
    lines: [
      'let graph = Graph::<AgentState>::new()',
      '    .add_node("plan", planner)',
      '    .add_node("act", executor)',
      '    .checkpointer(SqliteSaver::new(path))',
      '    .compile()?;',
    ],
    highlight: 3,
    calloutLabel: '02 · GRAPH',
    calloutTitle: 'Pregel-style supersteps, with time-travel.',
    calloutBody:
      'Every superstep is checkpointed. Pause for human approval, replay from any point, or fork a new branch from a past state — all through the same SqliteSaver.',
  },
  {
    key: 'rag',
    tab: 'RAG',
    file: 'rag.rs',
    lines: [
      'let retriever = VectorStore::in_memory()',
      '    .index(docs, embedder).await?',
      '    .as_retriever(TopK::new(5));',
      '',
      'let chain = retriever.pipe(prompt).pipe(model);',
    ],
    highlight: 4,
    calloutLabel: '03 · RAG',
    calloutTitle: 'A retriever is a Runnable, too.',
    calloutBody:
      'Pipe it straight into a prompt and a model with the same .pipe() every other primitive uses. No separate RAG-specific API to learn.',
  },
  {
    key: 'middleware',
    tab: 'Middleware',
    file: 'client.rs',
    lines: [
      'let client = Client::from_env()?',
      '    .with_max_retries(3)',
      '    .with_timeout(Duration::from_secs(30))',
      '    .with_rate_limit(TokenBucket::new(60, 10));',
    ],
    highlight: 1,
    calloutLabel: '04 · MIDDLEWARE',
    calloutTitle: 'Retry, cache, rate-limit — all wrappers.',
    calloutBody:
      'with_max_retries returns Retry<Self>, which is itself a Runnable. Wrappers compose, so they work on a client, an agent, or a single tool call identically.',
  },
];

const TERMINAL_LINES = [
  { text: '$ cargo run', dim: false },
  { text: '   Compiling cognis v0.3.2', dim: true },
  { text: '    Finished dev profile in 2.14s', dim: true },
  { text: '     Running `target/debug/agent`', dim: true },
  { text: '', dim: false },
  { text: '> What is 23 * 17 + 4?', dim: false },
  { text: '  [tool] calculator({"expr":"23*17+4"})', dim: true },
  { text: '  [tool] → 395', dim: true },
  { text: '  23 * 17 + 4 = 395', dim: false },
];

function Panel({
  title,
  className,
  children,
}: {
  title?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className={`rounded-lg border flex flex-col ${className ?? ''}`}
      style={{ borderColor: 'var(--color-fd-border)', background: 'var(--color-fd-card)' }}
    >
      {title && (
        <div
          className="px-4 py-2.5 text-xs border-b"
          style={{ borderColor: 'var(--color-fd-border)', color: 'var(--color-fd-muted-foreground)' }}
        >
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

export default function HomePage() {
  const [active, setActive] = useState(0);
  const pillar = PILLARS[active];

  return (
    <main className="flex-1">
      {/* hero: centered, shadcn-style */}
      <section className="mx-auto max-w-3xl px-6 pt-20 pb-14 text-center flex flex-col items-center">
        <Link
          href="/docs/reference/changelog"
          className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs"
          style={{ borderColor: 'var(--color-fd-border)', color: 'var(--color-fd-muted-foreground)' }}
        >
          v0.3 — DedupVectorStore, LLM extractors, fact extraction
          <span>→</span>
        </Link>

        <h1 className="mt-6 text-5xl md:text-6xl font-bold tracking-tight leading-[1.08]">
          Build LLM agents in Rust.
          <br />
          <span style={{ color: 'var(--color-fd-muted-foreground)' }}>Types checked at compile time.</span>
        </h1>

        <p className="mt-6 max-w-xl text-lg" style={{ color: 'var(--color-fd-muted-foreground)' }}>
          Typed Runnable pipelines, an agent loop, a stateful graph engine, and RAG
          primitives — built on Rust&apos;s type system, not around it.
        </p>

        <div className="mt-6 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-sm">
          <span className="inline-flex items-center gap-2">
            <CheckIcon /> Compile-time tool schemas
          </span>
          <span className="inline-flex items-center gap-2">
            <CheckIcon /> Feature-gated providers
          </span>
          <span className="inline-flex items-center gap-2">
            <CheckIcon /> One crate, four layers
          </span>
        </div>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
          <a
            href="#"
            className="inline-flex items-center gap-2 rounded-md px-5 py-2.5 text-sm font-medium"
            style={{ background: 'var(--color-fd-primary)', color: 'var(--color-fd-primary-foreground)' }}
          >
            cargo add cognis
          </a>
          <Link
            href="/docs"
            className="inline-flex items-center gap-2 rounded-md border px-5 py-2.5 text-sm font-medium"
            style={{ borderColor: 'var(--color-fd-border)' }}
          >
            Read the docs
          </Link>
        </div>

        <p className="mt-6 text-xs" style={{ color: 'var(--color-fd-muted-foreground)' }}>
          v0.3.2 · MIT license · 6 providers · 6 vector stores · rust 1.75+
        </p>
      </section>

      {/* bento proof grid */}
      <section className="mx-auto max-w-5xl px-6 pb-24">
        <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
          {/* big tabbed demo */}
          <Panel className="md:col-span-4 md:row-span-2 overflow-hidden">
            <div className="flex items-center gap-1 border-b px-2" style={{ borderColor: 'var(--color-fd-border)' }}>
              {PILLARS.map((p, i) => (
                <button
                  key={p.key}
                  onClick={() => setActive(i)}
                  className="px-4 py-3 text-sm"
                  style={{
                    color: i === active ? 'var(--color-fd-foreground)' : 'var(--color-fd-muted-foreground)',
                    borderBottom: i === active ? '2px solid var(--color-fd-primary)' : '2px solid transparent',
                  }}
                >
                  {p.tab}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2 px-4 py-2 text-xs" style={{ color: 'var(--color-fd-muted-foreground)' }}>
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: '#e8604a' }} />
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: '#e8b74a' }} />
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: '#6fbf73' }} />
              <span className="ml-2 font-mono">{pillar.file}</span>
            </div>
            <pre className="px-6 py-5 text-sm font-mono overflow-x-auto flex-1">
              {pillar.lines.map((line, i) => (
                <div
                  key={i}
                  className="px-2 -mx-2 rounded"
                  style={{ background: i === pillar.highlight ? 'var(--color-fd-accent)' : 'transparent' }}
                >
                  {line || ' '}
                </div>
              ))}
            </pre>
            <div className="border-t px-6 py-5" style={{ borderColor: 'var(--color-fd-border)' }}>
              <p className="text-xs font-mono" style={{ color: 'var(--color-fd-muted-foreground)' }}>
                {pillar.calloutLabel}
              </p>
              <p className="mt-1 font-semibold">{pillar.calloutTitle}</p>
              <p className="mt-1 text-sm" style={{ color: 'var(--color-fd-muted-foreground)' }}>
                {pillar.calloutBody}
              </p>
            </div>
          </Panel>

          {/* terminal output proof */}
          <Panel title="terminal" className="md:col-span-2 md:row-span-2">
            <pre className="px-5 py-4 text-xs font-mono flex-1 overflow-x-auto">
              {TERMINAL_LINES.map((l, i) => (
                <div key={i} style={{ color: l.dim ? 'var(--color-fd-muted-foreground)' : 'var(--color-fd-foreground)' }}>
                  {l.text || ' '}
                </div>
              ))}
            </pre>
          </Panel>

          {/* works with */}
          <Panel className="md:col-span-2 p-5">
            <p className="text-xs" style={{ color: 'var(--color-fd-muted-foreground)' }}>
              Works with
            </p>
            <p className="mt-2 text-sm font-mono">
              openai · anthropic · google
              <br />
              ollama · azure · openrouter
            </p>
          </Panel>

          {/* feature: compile time */}
          <Panel className="md:col-span-2 p-5">
            <p className="font-semibold text-sm">Compile-time guarantees</p>
            <p className="mt-1 text-sm" style={{ color: 'var(--color-fd-muted-foreground)' }}>
              Tool schemas and graph state transitions are checked before your code runs.
            </p>
          </Panel>

          {/* feature: feature-gated */}
          <Panel className="md:col-span-2 p-5">
            <p className="font-semibold text-sm">One umbrella, full stack</p>
            <p className="mt-1 text-sm" style={{ color: 'var(--color-fd-muted-foreground)' }}>
              cognis re-exports the foundation, LLM, RAG, and graph layers behind one prelude.
            </p>
          </Panel>
        </div>
      </section>
    </main>
  );
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path
        d="M3 8.5L6.2 11.5L13 4.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
