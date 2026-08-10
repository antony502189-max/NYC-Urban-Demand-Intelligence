def test_core_package_imports() -> None:
    import nyc_demand  # noqa: F401
    import nyc_demand.api.app  # noqa: F401
    import nyc_demand.data.aggregate  # noqa: F401
    import nyc_demand.features.builder  # noqa: F401
    import nyc_demand.models.backtest  # noqa: F401
    import nyc_demand.models.lightgbm_model  # noqa: F401
