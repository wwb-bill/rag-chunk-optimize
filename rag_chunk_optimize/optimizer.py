from rag_chunk_optimize.types import ChunkConfig, TrialResult, OptimizeReport
from rag_chunk_optimize.chunker import chunk_text

def optimize(documents: dict[str,str], queries: list[str], strategies=None, sizes=None, overlaps=None) -> OptimizeReport:
    strategies = strategies or ["sentence","paragraph","fixed"]
    sizes = sizes or [200,500,800,1200]
    overlaps = overlaps or [0,50,100]
    trials: list[TrialResult] = []
    for st in strategies:
        for sz in sizes:
            for ov in overlaps:
                if ov >= sz: continue
                cfg = ChunkConfig(strategy=st, size=sz, overlap=ov)
                trials.append(_eval(cfg, documents, queries))
    trials.sort(key=lambda t: t.score, reverse=True)
    return OptimizeReport(trials=trials, best=trials[0] if trials else None, total_tested=len(trials))

def _eval(cfg: ChunkConfig, docs: dict[str,str], qs: list[str]) -> TrialResult:
    all_c: list[str] = []
    for t in docs.values(): all_c.extend(chunk_text(t, cfg))
    if not all_c: return TrialResult(config=cfg)
    hits=0; covered:set[int]=set()
    for q in qs:
        qw=set(q.lower().split())
        if not qw: continue
        for i,c in enumerate(all_c):
            if qw & set(c.lower().split()): hits+=1; covered.add(i)
    n=max(len(qs),1)
    avg_h=hits/n
    cov=len(covered)/max(len(all_c),1)
    hit_s=min(1.0, avg_h/max(len(all_c)*0.5,1))
    cnt_p=min(1.0,20/max(len(all_c),1))
    return TrialResult(config=cfg,chunk_count=len(all_c),avg_hits=round(avg_h,2),coverage=round(cov,4),score=round(cov*0.6+hit_s*0.3+cnt_p*0.1,4))
