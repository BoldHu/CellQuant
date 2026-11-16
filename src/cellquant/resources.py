"""Sample process RSS during jobs, including child processes, without blocking."""
import threading
import time
import psutil


class ResourceMonitor:
    def __init__(self,interval=.2):
        self.interval=interval;self.peak_rss_bytes=0;self.samples=0;self._stop=threading.Event()
        self._process=psutil.Process();self._thread=None
    def _sample(self):
        total=0
        for process in [self._process,*self._process.children(recursive=True)]:
            try:total+=process.memory_info().rss
            except psutil.NoSuchProcess:continue
        self.peak_rss_bytes=max(self.peak_rss_bytes,total);self.samples+=1
    def _run(self):
        while not self._stop.wait(self.interval):self._sample()
    def start(self):
        self.started=time.perf_counter();self._sample()
        self._thread=threading.Thread(target=self._run,daemon=True);self._thread.start();return self
    def stop(self):
        self._stop.set()
        if self._thread:self._thread.join(timeout=1)
        self._sample()
        return {'sampled_peak_process_tree_rss_bytes':self.peak_rss_bytes,'rss_sample_count':self.samples,
                'rss_sampling_interval_seconds':self.interval,'wall_seconds':time.perf_counter()-self.started,
                'scope':'Sampled process-tree resident memory; short peaks between samples may be missed'}
