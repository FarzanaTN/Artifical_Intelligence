import time
import tracemalloc

def run_with_metrics(func):
    tracemalloc.start()

    start = time.perf_counter()
    result = func()
    end = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "result": result,
        "time": end - start,
        "memory": peak / 1024
    }