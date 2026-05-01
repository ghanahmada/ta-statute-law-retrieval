"use client";

import { useState, useEffect, useMemo, useRef } from "react";

/* ── types ── */

interface Highlight { start: number; end: number; token: string; intensity: number; score: number }
interface PerToken { token: string; contribution: number; idf: number; tf: number; in_doc: boolean }
interface BM25RankedDoc {
  rank: number; doc_id: string; title: string; bm25_score: number; is_relevant: boolean;
  per_token: PerToken[]; doc_highlights: Highlight[]; query_highlights: Highlight[];
  beyond_topk?: boolean;
}
interface BM25QueryDetail {
  qid: string; query_text: string; case_name: string; n_gt: number;
  gt_doc_ids: string[]; gt_ranks: Record<string, number>; ranked_docs: BM25RankedDoc[];
}

interface GNNRankedDoc {
  rank: number | null; doc_id: string; title: string;
  score: number; gnn_score: number; bm25_score: number;
  is_relevant: boolean; beyond_topk?: boolean;
}
interface GNNQueryDetail {
  qid: string; query_text: string; case_name: string; n_gt: number;
  gt_doc_ids: string[]; gt_ranks: Record<string, number>; ranked_docs: GNNRankedDoc[];
}

interface IndexEntry {
  qid: string; preview: string; case_name: string; n_gt: number;
  has_hit: boolean; best_gt_rank: number | null; worst_gt_rank: number | null;
}
interface BM25Summary {
  n_queries: number; n_gt_docs: number; gt_in_top10: number; gt_in_top25: number;
  recall_at_10: number; recall_at_25: number; queries_zero_hit_at_10: number;
  gt_rank_median: number; gt_rank_mean: number;
}
interface GNNSummary {
  dataset: string; model: string; alpha: number;
  n_queries: number; n_gt_docs: number; gt_in_top10: number; gt_in_top25: number;
  recall_at_10: number; recall_at_25: number; mrr_at_10: number; hit_rate: number;
  queries_zero_hit_at_10: number; gnn_helped: number;
  gt_rank_median: number; gt_rank_mean: number;
}
interface GNNIndex { summary: GNNSummary; queries: IndexEntry[] }

interface Corpus { [id: string]: { title: string; text: string } }

type SortMode = "default" | "best_gt" | "worst_gt";
type ViewTab = "bm25" | "structgnn";

/* ── highlighted text ── */

function HL({ text, highlights }: { text: string; highlights: Highlight[] }) {
  if (!highlights?.length) return <>{text}</>;
  const parts: React.ReactNode[] = [];
  let cur = 0;
  for (const h of [...highlights].sort((a, b) => a.start - b.start)) {
    if (h.start < cur) continue;
    if (h.start > cur) parts.push(<span key={`p${cur}`}>{text.slice(cur, h.start)}</span>);
    const op = Math.max(0.12, Math.min(0.65, h.intensity * 0.65));
    parts.push(
      <mark key={`h${h.start}`} className="rounded-sm px-0.5 -mx-0.5"
        style={{ backgroundColor: `rgba(212,160,23,${op})`, color: "inherit" }}
        title={`"${h.token}" score=${h.score.toFixed(3)}`}>
        {text.slice(h.start, h.end)}
      </mark>
    );
    cur = h.end;
  }
  if (cur < text.length) parts.push(<span key={`p${cur}`}>{text.slice(cur)}</span>);
  return <>{parts}</>;
}

/* ── stat ── */

function Stat({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="px-5 py-2.5">
      <div className="text-[10px] uppercase tracking-[0.14em] text-[var(--color-text-dim)] font-medium">{label}</div>
      <div className="text-lg font-semibold font-[family-name:var(--font-mono)] mt-0.5">{value}</div>
      {sub && <div className="text-[10px] text-[var(--color-text-muted)] mt-0.5">{sub}</div>}
    </div>
  );
}

/* ── rank badge ── */

function RankBadge({ rank }: { rank: number }) {
  const cls = rank <= 10
    ? "bg-[var(--color-gt-bg)] text-[var(--color-gt-green)] border-[var(--color-gt-border)]"
    : rank <= 50 ? "bg-amber-950/30 text-amber-400 border-amber-800/40"
    : "bg-red-950/20 text-red-400 border-red-800/30";
  return <span className={`inline-flex items-center text-[11px] font-[family-name:var(--font-mono)] font-medium rounded border px-1.5 py-0.5 ${cls}`}>#{rank}</span>;
}

/* ── score bar ── */

function ScoreBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
  return (
    <div className="flex items-center gap-2">
      <span className="text-[9px] font-[family-name:var(--font-mono)] text-[var(--color-text-dim)] w-11 text-right shrink-0">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-[var(--color-border-subtle)] overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-[10px] font-[family-name:var(--font-mono)] text-[var(--color-text-muted)] tabular-nums w-12 text-right">{value.toFixed(2)}</span>
    </div>
  );
}

/* ── BM25 doc card ── */

function BM25DocCard({ doc, corpus, maxScore }: { doc: BM25RankedDoc; corpus: Corpus; maxScore: number }) {
  const [open, setOpen] = useState(false);
  const text = corpus[doc.doc_id]?.text || "";
  const pct = maxScore > 0 ? Math.min(100, (doc.bm25_score / maxScore) * 100) : 0;
  const pills = doc.per_token.filter((t) => t.contribution > 0);
  const gt = doc.is_relevant;

  return (
    <div onClick={() => setOpen(!open)}
      className={`rounded-r-lg cursor-pointer transition-colors ${
        gt ? "border-l-2 border-l-[var(--color-gt-green)] bg-[rgba(5,46,22,0.25)]"
           : "border-l-2 border-l-transparent bg-[var(--color-surface-raised)]"
      } hover:bg-[var(--color-surface-hover)]`}>
      <div className="px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="text-sm font-[family-name:var(--font-mono)] font-medium text-[var(--color-text-dim)] w-12 text-right shrink-0">{doc.rank}</span>
          <span className="text-sm font-semibold">{doc.title}</span>
          {gt && <span className="text-[9px] uppercase tracking-widest font-bold text-[var(--color-gt-green)] bg-[var(--color-gt-bg)] border border-[var(--color-gt-border)] rounded px-1.5 py-0.5">GT</span>}
          <div className="ml-auto flex items-center gap-2 min-w-[130px]">
            <div className="flex-1 h-1.5 rounded-full bg-[var(--color-border-subtle)] overflow-hidden">
              <div className="h-full rounded-full bg-[var(--color-score-bar)] transition-all" style={{ width: `${pct}%` }} />
            </div>
            <span className="text-[10px] font-[family-name:var(--font-mono)] text-[var(--color-text-muted)] tabular-nums w-10 text-right">{doc.bm25_score.toFixed(1)}</span>
          </div>
        </div>
        <div className="mt-2 ml-15">
          <div className={`text-[13px] leading-[1.7] font-[family-name:var(--font-mono)] text-[var(--color-text-muted)] ${open ? "" : "line-clamp-3"}`}>
            <HL text={text} highlights={doc.doc_highlights} />
          </div>
          {pills.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {pills.slice(0, open ? 20 : 6).map((t) => (
                <span key={t.token}
                  className="inline-flex items-center gap-1 text-[10px] font-[family-name:var(--font-mono)] rounded-full px-2 py-0.5 border"
                  style={{
                    backgroundColor: `rgba(212,160,23,${Math.min(0.18, t.contribution / 12)})`,
                    borderColor: `rgba(212,160,23,${Math.min(0.35, t.contribution / 8)})`,
                    color: "var(--color-amber-glow)",
                  }}
                  title={`IDF=${t.idf.toFixed(2)} TF=${t.tf}`}>
                  {t.token} <span className="opacity-50">{t.contribution.toFixed(1)}</span>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── GNN doc card ── */

function GNNDocCard({ doc, corpus, maxScores, alpha }: {
  doc: GNNRankedDoc; corpus: Corpus;
  maxScores: { blended: number; gnn: number; bm25: number };
  alpha: number;
}) {
  const [open, setOpen] = useState(false);
  const text = corpus[doc.doc_id]?.text || "";
  const gt = doc.is_relevant;
  const isBeyond = doc.beyond_topk;
  const gnnDominant = Math.abs(doc.gnn_score) > Math.abs(doc.bm25_score);

  return (
    <div onClick={() => setOpen(!open)}
      className={`rounded-r-lg cursor-pointer transition-colors ${
        gt ? "border-l-2 border-l-[var(--color-gt-green)] bg-[rgba(5,46,22,0.25)]"
           : "border-l-2 border-l-transparent bg-[var(--color-surface-raised)]"
      } hover:bg-[var(--color-surface-hover)]`}>
      <div className="px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="text-sm font-[family-name:var(--font-mono)] font-medium text-[var(--color-text-dim)] w-12 text-right shrink-0">
            {isBeyond ? "\u2014" : doc.rank}
          </span>
          <span className="text-sm font-semibold">{doc.title}</span>
          {gt && <span className="text-[9px] uppercase tracking-widest font-bold text-[var(--color-gt-green)] bg-[var(--color-gt-bg)] border border-[var(--color-gt-border)] rounded px-1.5 py-0.5">GT</span>}
          {isBeyond && <span className="text-[9px] uppercase tracking-widest font-medium text-amber-400 bg-amber-950/30 border border-amber-800/40 rounded px-1.5 py-0.5">beyond top-k</span>}
          <span className={`ml-auto text-[9px] font-[family-name:var(--font-mono)] px-1.5 py-0.5 rounded ${
            gnnDominant ? "text-purple-400 bg-purple-950/30" : "text-amber-400 bg-amber-950/30"
          }`}>
            {gnnDominant ? "GNN-driven" : "BM25-driven"}
          </span>
        </div>

        <div className="mt-2 ml-15 space-y-1">
          <ScoreBar label="blend" value={doc.score} max={maxScores.blended} color="var(--color-blended-blue)" />
          <ScoreBar label="gnn" value={doc.gnn_score} max={maxScores.gnn} color="var(--color-gnn-purple)" />
          <ScoreBar label="bm25" value={doc.bm25_score} max={maxScores.bm25} color="var(--color-bm25-amber)" />
        </div>

        {open && (
          <div className="mt-2 ml-15 p-2.5 rounded bg-[var(--color-surface)] border border-[var(--color-border-subtle)]">
            <div className="text-[10px] font-[family-name:var(--font-mono)] text-[var(--color-text-muted)] space-y-0.5">
              <div>blended = {alpha.toFixed(1)} x gnn_norm + {(1 - alpha).toFixed(1)} x bm25</div>
              <div>       = {alpha.toFixed(1)} x {doc.gnn_score.toFixed(3)} + {(1 - alpha).toFixed(1)} x {doc.bm25_score.toFixed(3)}</div>
              <div>       = {doc.score.toFixed(3)}</div>
            </div>
          </div>
        )}

        <div className="mt-2 ml-15">
          <div className={`text-[13px] leading-[1.7] font-[family-name:var(--font-mono)] text-[var(--color-text-muted)] ${open ? "" : "line-clamp-3"}`}>
            {text}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── main ── */

export default function Home() {
  const [bm25Index, setBm25Index] = useState<{ summary: BM25Summary; queries: IndexEntry[] } | null>(null);
  const [gnnIndex, setGnnIndex] = useState<GNNIndex | null>(null);
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [bm25Detail, setBm25Detail] = useState<BM25QueryDetail | null>(null);
  const [gnnDetail, setGnnDetail] = useState<GNNQueryDetail | null>(null);
  const [loadingBm25, setLoadingBm25] = useState(false);
  const [loadingGnn, setLoadingGnn] = useState(false);
  const [selQid, setSelQid] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("default");
  const [light, setLight] = useState(false);
  const [tab, setTab] = useState<ViewTab>("bm25");
  const bm25Cache = useRef<Record<string, BM25QueryDetail>>({});
  const gnnCache = useRef<Record<string, GNNQueryDetail>>({});

  useEffect(() => {
    document.documentElement.classList.toggle("light", light);
  }, [light]);

  useEffect(() => {
    Promise.all([
      fetch("/index.json").then((r) => r.json()),
      fetch("/corpus.json").then((r) => r.json()),
      fetch("/structgnn/index.json").then((r) => r.json()),
    ]).then(([bm25Idx, corp, gnnIdx]) => {
      setBm25Index(bm25Idx);
      setCorpus(corp);
      setGnnIndex(gnnIdx);
      if (bm25Idx.queries.length > 0) setSelQid(bm25Idx.queries[0].qid);
    });
  }, []);

  useEffect(() => {
    if (!selQid) return;

    if (bm25Cache.current[selQid]) {
      setBm25Detail(bm25Cache.current[selQid]);
    } else {
      setLoadingBm25(true);
      fetch(`/queries/${selQid}.json`)
        .then((r) => r.json())
        .then((d) => { bm25Cache.current[selQid] = d; setBm25Detail(d); setLoadingBm25(false); });
    }

    if (gnnCache.current[selQid]) {
      setGnnDetail(gnnCache.current[selQid]);
    } else {
      setLoadingGnn(true);
      fetch(`/structgnn/queries/${selQid}.json`)
        .then((r) => r.json())
        .then((d) => { gnnCache.current[selQid] = d; setGnnDetail(d); setLoadingGnn(false); })
        .catch(() => { setGnnDetail(null); setLoadingGnn(false); });
    }
  }, [selQid]);

  const filtered = useMemo(() => {
    if (!bm25Index) return [];
    let list = bm25Index.queries;
    if (search) {
      const s = search.toLowerCase();
      list = list.filter((q) => q.qid.includes(s) || q.preview.toLowerCase().includes(s));
    }
    if (sortMode === "best_gt") {
      list = [...list].sort((a, b) => (a.best_gt_rank ?? 9999) - (b.best_gt_rank ?? 9999));
    } else if (sortMode === "worst_gt") {
      list = [...list].sort((a, b) => (b.worst_gt_rank ?? 0) - (a.worst_gt_rank ?? 0));
    }
    return list;
  }, [bm25Index, search, sortMode]);

  const bm25MaxScore = useMemo(() => {
    if (!bm25Detail) return 1;
    return Math.max(...bm25Detail.ranked_docs.map((d) => d.bm25_score), 1);
  }, [bm25Detail]);

  const gnnMaxScores = useMemo(() => {
    if (!gnnDetail) return { blended: 1, gnn: 1, bm25: 1 };
    const docs = gnnDetail.ranked_docs;
    return {
      blended: Math.max(...docs.map((d) => Math.abs(d.score)), 1),
      gnn: Math.max(...docs.map((d) => Math.abs(d.gnn_score)), 1),
      bm25: Math.max(...docs.map((d) => Math.abs(d.bm25_score)), 1),
    };
  }, [gnnDetail]);

  const bm25Split = useMemo(() => {
    if (!bm25Detail) return { topkDocs: [], beyondDocs: [] };
    return {
      topkDocs: bm25Detail.ranked_docs.filter((d) => !d.beyond_topk),
      beyondDocs: bm25Detail.ranked_docs.filter((d) => d.beyond_topk),
    };
  }, [bm25Detail]);

  const gnnSplit = useMemo(() => {
    if (!gnnDetail) return { topkDocs: [], beyondDocs: [] };
    return {
      topkDocs: gnnDetail.ranked_docs.filter((d) => !d.beyond_topk),
      beyondDocs: gnnDetail.ranked_docs.filter((d) => d.beyond_topk),
    };
  }, [gnnDetail]);

  if (!bm25Index || !corpus) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-7 h-7 border-2 border-[var(--color-border)] border-t-[var(--color-amber-hl)] rounded-full animate-spin mx-auto" />
          <p className="mt-3 text-sm text-[var(--color-text-muted)]">Loading...</p>
        </div>
      </div>
    );
  }

  const bm25Sm = bm25Index.summary;
  const gnnSm = gnnIndex?.summary;

  const detail = tab === "bm25" ? bm25Detail : gnnDetail;
  const loading = tab === "bm25" ? loadingBm25 : loadingGnn;

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* header */}
      <header className="shrink-0 border-b border-[var(--color-border)]">
        <div className="px-5 py-2 flex items-center gap-4">
          <div>
            <h1 className="text-sm font-semibold tracking-tight">Retrieval Analysis</h1>
            <p className="text-[10px] text-[var(--color-text-dim)]">KUHPerdata Humanized</p>
          </div>
          <button onClick={() => setLight(!light)}
            className="text-[10px] px-2.5 py-1 rounded border border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors">
            {light ? "Dark" : "Light"}
          </button>
          <div className="ml-auto flex divide-x divide-[var(--color-border-subtle)]">
            {tab === "bm25" ? (
              <>
                <Stat label="Queries" value={bm25Sm.n_queries} sub="max_relevant=5" />
                <Stat label="Recall@10" value={(bm25Sm.recall_at_10 * 100).toFixed(1) + "%"} sub={`${bm25Sm.gt_in_top10} of ${bm25Sm.n_gt_docs} GT docs`} />
                <Stat label="Recall@25" value={(bm25Sm.recall_at_25 * 100).toFixed(1) + "%"} sub={`${bm25Sm.gt_in_top25} of ${bm25Sm.n_gt_docs} GT docs`} />
                <Stat label="Zero-hit @10" value={bm25Sm.queries_zero_hit_at_10} sub={`${((bm25Sm.queries_zero_hit_at_10 / bm25Sm.n_queries) * 100).toFixed(0)}% of queries`} />
                <Stat label="GT Rank Med." value={bm25Sm.gt_rank_median.toFixed(0)} sub={`mean ${bm25Sm.gt_rank_mean.toFixed(0)}`} />
              </>
            ) : gnnSm ? (
              <>
                <Stat label="MRR@10" value={gnnSm.mrr_at_10.toFixed(3)} />
                <Stat label="Hit Rate" value={(gnnSm.hit_rate * 100).toFixed(1) + "%"} sub={`${gnnSm.n_queries - gnnSm.queries_zero_hit_at_10}/${gnnSm.n_queries} queries`} />
                <Stat label="Recall@10" value={(gnnSm.recall_at_10 * 100).toFixed(1) + "%"} sub={`${gnnSm.gt_in_top10} of ${gnnSm.n_gt_docs} GT docs`} />
                <Stat label="Recall@25" value={(gnnSm.recall_at_25 * 100).toFixed(1) + "%"} sub={`${gnnSm.gt_in_top25} of ${gnnSm.n_gt_docs} GT docs`} />
                <Stat label="Alpha" value={gnnSm.alpha.toFixed(1)} sub={gnnSm.model} />
              </>
            ) : null}
          </div>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* sidebar */}
        <div className="w-[360px] shrink-0 border-r border-[var(--color-border)] flex flex-col">
          <div className="p-2.5 border-b border-[var(--color-border-subtle)]">
            <input type="text" placeholder="Search queries..." value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-[var(--color-surface-hover)] text-sm rounded-md px-3 py-1.5 border border-[var(--color-border)] focus:outline-none focus:border-[var(--color-amber-hl)] placeholder:text-[var(--color-text-dim)]" />
            <div className="flex items-center gap-1.5 mt-1.5 px-0.5">
              <span className="text-[9px] text-[var(--color-text-dim)]">{filtered.length} queries — sort:</span>
              {(["default", "best_gt", "worst_gt"] as SortMode[]).map((mode) => (
                <button key={mode} onClick={() => setSortMode(mode)}
                  className={`text-[9px] px-1.5 py-0.5 rounded transition-colors ${
                    sortMode === mode
                      ? "bg-[var(--color-amber-hl)] text-[var(--color-surface)]"
                      : "text-[var(--color-text-dim)] hover:text-[var(--color-text-muted)]"
                  }`}>
                  {mode === "default" ? "ID" : mode === "best_gt" ? "Best GT rank" : "Worst GT rank"}
                </button>
              ))}
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            {filtered.map((q) => {
              const active = q.qid === selQid;
              return (
                <button key={q.qid} onClick={() => setSelQid(q.qid)}
                  className={`w-full text-left px-3 py-2 border-b border-[var(--color-border-subtle)] transition-colors ${
                    active ? "bg-[var(--color-surface-hover)] border-l-2 border-l-[var(--color-amber-hl)]"
                           : "border-l-2 border-l-transparent hover:bg-[var(--color-surface-raised)]"
                  }`}>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-[family-name:var(--font-mono)] text-[var(--color-text-dim)]">{q.qid}</span>
                    <span className={`text-[8px] font-bold rounded px-1 py-0.5 ${q.has_hit ? "bg-[var(--color-gt-bg)] text-[var(--color-gt-green)]" : "bg-red-950/30 text-red-400"}`}>
                      {q.has_hit ? "HIT" : "MISS"}
                    </span>
                    {q.best_gt_rank != null && (
                      <span className="text-[9px] font-[family-name:var(--font-mono)] text-[var(--color-text-dim)] ml-auto">
                        GT best:<span className={q.best_gt_rank <= 10 ? "text-[var(--color-gt-green)]" : q.best_gt_rank <= 50 ? "text-amber-400" : "text-red-400"}> #{q.best_gt_rank}</span>
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-[var(--color-text-muted)] mt-0.5 line-clamp-2 leading-relaxed">{q.preview}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* detail */}
        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center h-32">
              <div className="w-5 h-5 border-2 border-[var(--color-border)] border-t-[var(--color-amber-hl)] rounded-full animate-spin" />
            </div>
          ) : detail ? (
            <div className="max-w-4xl mx-auto px-6 py-5">
              {/* query */}
              <div className="mb-5">
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="text-xs font-[family-name:var(--font-mono)] text-[var(--color-text-dim)]">{detail.qid}</span>
                  <span className="text-[10px] text-[var(--color-text-dim)] truncate">{detail.case_name}</span>
                </div>
                <div className="text-[15px] leading-relaxed font-[family-name:var(--font-mono)] bg-[var(--color-surface-raised)] rounded-lg p-4 border border-[var(--color-border)]">
                  {tab === "bm25" && bm25Detail
                    ? <HL text={detail.query_text} highlights={(bm25Detail.ranked_docs[0] as BM25RankedDoc)?.query_highlights || []} />
                    : detail.query_text}
                </div>
              </div>

              {/* GT */}
              <div className="mb-5 p-3.5 rounded-lg bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
                <div className="text-[10px] uppercase tracking-[0.14em] text-[var(--color-text-dim)] font-medium mb-2">
                  Ground Truth ({detail.n_gt} articles)
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1.5">
                  {detail.gt_doc_ids.map((did) => {
                    const bm25Rank = bm25Detail?.gt_ranks[did];
                    const gnnRank = gnnDetail?.gt_ranks[did];
                    return (
                      <div key={did} className="flex items-center gap-1.5">
                        <span className="text-sm font-[family-name:var(--font-mono)] font-medium">{corpus[did]?.title || `Pasal ${did}`}</span>
                        {bm25Rank != null && (
                          <span className="text-[9px] font-[family-name:var(--font-mono)] text-[var(--color-text-dim)]">
                            BM25:<span className={bm25Rank <= 10 ? "text-[var(--color-gt-green)]" : bm25Rank <= 50 ? "text-amber-400" : "text-red-400"}>#{bm25Rank}</span>
                          </span>
                        )}
                        {gnnRank != null && (
                          <span className="text-[9px] font-[family-name:var(--font-mono)] text-[var(--color-text-dim)]">
                            GNN:<span className={gnnRank <= 10 ? "text-[var(--color-gt-green)]" : gnnRank <= 50 ? "text-amber-400" : "text-red-400"}>#{gnnRank}</span>
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* tabs */}
              <div className="mb-3 flex items-center gap-2">
                <button onClick={() => setTab("bm25")}
                  className={`text-[11px] font-medium px-3 py-1.5 rounded-t border transition-colors ${
                    tab === "bm25"
                      ? "border-[var(--color-amber-hl)]/50 bg-amber-950/30 text-[var(--color-amber-glow)] border-b-transparent"
                      : "border-transparent text-[var(--color-text-dim)] hover:text-[var(--color-text-muted)]"
                  }`}>
                  BM25 Ranking
                </button>
                <button onClick={() => setTab("structgnn")}
                  className={`text-[11px] font-medium px-3 py-1.5 rounded-t border transition-colors ${
                    tab === "structgnn"
                      ? "border-purple-700/50 bg-purple-950/30 text-purple-400 border-b-transparent"
                      : "border-transparent text-[var(--color-text-dim)] hover:text-[var(--color-text-muted)]"
                  }`}>
                  StructGNN Ranking
                </button>
                {tab === "structgnn" && (
                  <div className="flex items-center gap-3 ml-auto">
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-1.5 rounded-full bg-[var(--color-blended-blue)]" />
                      <span className="text-[9px] text-[var(--color-text-dim)]">Blended</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-1.5 rounded-full bg-[var(--color-gnn-purple)]" />
                      <span className="text-[9px] text-[var(--color-text-dim)]">GNN</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-1.5 rounded-full bg-[var(--color-bm25-amber)]" />
                      <span className="text-[9px] text-[var(--color-text-dim)]">BM25</span>
                    </div>
                  </div>
                )}
              </div>

              {/* BM25 tab content */}
              {tab === "bm25" && bm25Detail && (
                <>
                  <div className="space-y-0.5">
                    {bm25Split.topkDocs.map((doc) => (
                      <BM25DocCard key={doc.doc_id} doc={doc} corpus={corpus} maxScore={bm25MaxScore} />
                    ))}
                  </div>
                  {bm25Split.beyondDocs.length > 0 && (
                    <>
                      <div className="my-4 flex items-center gap-3">
                        <div className="flex-1 h-px bg-[var(--color-gt-border)]" />
                        <span className="text-[10px] uppercase tracking-[0.14em] text-[var(--color-gt-green)] font-medium">
                          GT beyond top-25 ({bm25Split.beyondDocs.length})
                        </span>
                        <div className="flex-1 h-px bg-[var(--color-gt-border)]" />
                      </div>
                      <div className="space-y-0.5">
                        {bm25Split.beyondDocs.map((doc) => (
                          <BM25DocCard key={doc.doc_id} doc={doc} corpus={corpus} maxScore={bm25MaxScore} />
                        ))}
                      </div>
                    </>
                  )}
                </>
              )}

              {/* StructGNN tab content */}
              {tab === "structgnn" && gnnDetail && gnnSm && (
                <>
                  <div className="space-y-0.5">
                    {gnnSplit.topkDocs.map((doc) => (
                      <GNNDocCard key={doc.doc_id} doc={doc} corpus={corpus} maxScores={gnnMaxScores} alpha={gnnSm.alpha} />
                    ))}
                  </div>
                  {gnnSplit.beyondDocs.length > 0 && (
                    <>
                      <div className="my-4 flex items-center gap-3">
                        <div className="flex-1 h-px bg-[var(--color-gt-border)]" />
                        <span className="text-[10px] uppercase tracking-[0.14em] text-[var(--color-gt-green)] font-medium">
                          GT beyond top-{gnnSplit.topkDocs.length} ({gnnSplit.beyondDocs.length})
                        </span>
                        <div className="flex-1 h-px bg-[var(--color-gt-border)]" />
                      </div>
                      <div className="space-y-0.5">
                        {gnnSplit.beyondDocs.map((doc) => (
                          <GNNDocCard key={doc.doc_id} doc={doc} corpus={corpus} maxScores={gnnMaxScores} alpha={gnnSm.alpha} />
                        ))}
                      </div>
                    </>
                  )}
                </>
              )}

              {tab === "structgnn" && !gnnDetail && !loadingGnn && (
                <div className="text-center py-8 text-[var(--color-text-dim)]">No StructGNN data for this query</div>
              )}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-[var(--color-text-dim)]">Select a query</div>
          )}
        </div>
      </div>
    </div>
  );
}
