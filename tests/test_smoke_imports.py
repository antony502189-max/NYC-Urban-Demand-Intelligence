def test_core_package_imports() -> None:
    import nyc_demand
    import nyc_demand.api.app
    import nyc_demand.data.aggregate
    import nyc_demand.features.builder
    import nyc_demand.models.backtest
    import nyc_demand.models.lightgbm_model  # noqa: F401
