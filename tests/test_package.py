import lottery


def test_version_comes_from_package_metadata():
    assert lottery.__version__ == "2.0.0"


def test_public_api_exports_resolve():
    for name in lottery.__all__:
        assert hasattr(lottery, name), name
