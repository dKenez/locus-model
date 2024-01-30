from tests import _PATH_DATA

def test_raw_data():
    assert (_PATH_DATA / "raw/LDoGI/shards").exists()
