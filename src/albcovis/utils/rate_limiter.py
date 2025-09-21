import threading, time

class RateLimiter:
    def __init__(self, rate_per_sec=1):
        self.lock = threading.Lock()
        self.min_interval = 1.0 / rate_per_sec
        self.last_time = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_time = time.time()

# Shared limiters for APIs
mb_limiter = RateLimiter(rate_per_sec=1)   # MusicBrainz → 1 request/sec
dg_limiter = RateLimiter(rate_per_sec=1)   # Discogs → 1 request/sec
