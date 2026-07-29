import sys, json, argparse
from rag_chunk_optimize.optimizer import optimize

def main(argv=None):
    if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    p = argparse.ArgumentParser(prog="rag-chunk-optimize")
    sub = p.add_subparsers(dest="cmd")
    p1 = sub.add_parser("run"); p1.add_argument("corpus"); p1.add_argument("queries"); p1.add_argument("--strategies",default="sentence,paragraph,fixed"); p1.add_argument("--sizes",default="200,500,800,1200"); p1.add_argument("--overlaps",default="0,50,100"); p1.add_argument("--json",action="store_true")
    args = p.parse_args(argv)
    try:
        if args.cmd=="run":
            with open(args.corpus,encoding="utf-8") as f: docs=json.load(f)
            qs=[]
            with open(args.queries,encoding="utf-8") as f:
                for l in f:
                    l=l.strip()
                    if not l: continue
                    qs.append(json.loads(l).get("text",json.loads(l).get("query","")))
            sts=[s.strip() for s in args.strategies.split(",")]
            szs=[int(x) for x in args.sizes.split(",")]
            ovs=[int(x) for x in args.overlaps.split(",")]
            r = optimize(docs,qs,sts,szs,ovs)
            if args.json: print(json.dumps({"best":{"strategy":r.best.config.strategy,"size":r.best.config.size,"overlap":r.best.config.overlap,"score":r.best.score} if r.best else None,"total":r.total_tested},indent=2))
            else: print(r.summary())
        else: p.print_help()
    except Exception as e: print(f"Error: {e}",file=sys.stderr); sys.exit(2)

if __name__=="__main__": main()
