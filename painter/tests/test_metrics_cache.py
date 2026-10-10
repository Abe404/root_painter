"""
Tests for reading and writing the metrics cache (metrics_cache.pkl).

The cache only saves recomputing metrics. A truncated cache file has been seen
in a real project, which crashed the metrics plot. An unreadable cache should
be treated as empty so it is rebuilt.
"""
import os
import sys
import pickle
import tempfile

test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src', 'main', 'python')
sys.path.insert(0, src_dir)

from plot_seg_metrics import load_metrics_cache, save_metrics_cache


def example_cache():
    return {f'img_{i:03d}.png': {'tp': i, 'fp': 0, 'fn': 0, 'tn': 100}
            for i in range(200)}


def test_missing_cache_is_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, 'metrics_cache.pkl')
        assert load_metrics_cache(cache_path) == {}


def test_save_then_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, 'metrics_cache.pkl')
        save_metrics_cache(example_cache(), cache_path)
        assert load_metrics_cache(cache_path) == example_cache()
        # the temporary file is replaced, not left behind.
        assert os.listdir(tmpdir) == ['metrics_cache.pkl']


def test_truncated_cache_is_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, 'metrics_cache.pkl')
        data = pickle.dumps(example_cache())
        with open(cache_path, 'wb') as cache_file:
            cache_file.write(data[:len(data) // 2])
        assert load_metrics_cache(cache_path) == {}


def test_empty_cache_file_is_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, 'metrics_cache.pkl')
        open(cache_path, 'wb').close()
        assert load_metrics_cache(cache_path) == {}
