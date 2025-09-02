# BBB/src/run_delta.py
import os
from .experiments import run_delta_parallel

def main() -> None:
    # set BBB_WORKERS to choose how many processes; omit or 0 -> auto
    workers = int(os.getenv("BBB_WORKERS", "0")) or None
    res = run_delta_parallel(workers=workers)
    print(f"overall_win_rate={res.overall_win_rate:.4f}")
    print(f"score={res.score:.4f}")
    for name, wr in res.per_strategy:
        print(f"{name}\t{wr:.4f}")

if __name__ == "__main__":
    main()

