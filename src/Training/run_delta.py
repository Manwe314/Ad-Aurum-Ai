# BBB/src/run_delta.py
import os
import time
from .experiments import run_delta_parallel

def main() -> None:
    # set BBB_WORKERS to choose how many processes; omit or 0 -> auto
    workers = int(os.getenv("BBB_WORKERS", "0")) or None
    t0 = time.time()
    res = run_delta_parallel(workers=workers)
    elapsed = time.time() - t0
    print(f"Delta experiment completed in {elapsed:.1f} seconds")
    print(f"overall_win_rate={res.overall_win_rate:.4f}")
    print(f"score={res.score:.4f}")
    for name, wr in res.per_strategy:
        print(f"{name}\t{wr:.4f}")

if __name__ == "__main__":
    main()

