def test_cpp_extension_import_smoke() -> None:
    import microalpha._cpp as cpp

    assert hasattr(cpp, "compute_features_series")
    assert callable(cpp.compute_features_series)