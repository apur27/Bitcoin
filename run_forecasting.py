import argparse
import os
import logging
from bitcoinML7 import UltimateBitcoinForecaster


def main():
    """Command-line interface for training and forecasting Bitcoin prices using
    UltimateBitcoinForecaster.

    Environment variables:
        TWELVE_API_KEY : API key for TwelveData (required for --fetch)
        FRED_API_KEY   : API key for FRED (optional, enables macro data)

    Typical usages:
        Fetch fresh data and train all models then predict 30 days ahead
            python run_forecasting.py --fetch --train --predict 30

        Train on previously prepared dataset only
            python run_forecasting.py --train

        Produce a 14-day ahead forecast using previously trained models
            python run_forecasting.py --predict 14
    """

    parser = argparse.ArgumentParser(
        description="Bitcoin Price Forecasting with UltimateBitcoinForecaster"
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch/refresh price, volume and exogenous data before any other step.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5000,
        help="Number of historical days to fetch when --fetch is used (max 5000).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train/optimise models on the prepared dataset.",
    )
    parser.add_argument(
        "--predict",
        type=int,
        metavar="N",
        help="Generate N-day ahead forecast using trained models.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Look-back window (in days) used for ML features.",
    )

    args = parser.parse_args()

    # Prepare logger for CLI run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    btc = UltimateBitcoinForecaster()

    # GPU/CPU info
    btc.check_gpu()

    # Read API keys from environment variables
    twelve_key = os.getenv("TWELVE_API_KEY")
    fred_key = os.getenv("FRED_API_KEY")
    if fred_key:
        btc.fred_api_key = fred_key

    if args.fetch:
        if not twelve_key:
            parser.error("--fetch requires TWELVE_API_KEY environment variable to be set.")
        btc.fetch_bitcoin_data(days=args.days, api_key=twelve_key)
        btc.fetch_volume_from_binance(days=args.days)
        btc.add_cycle_features()
        btc.calculate_indicators()
        btc.engineer_directional_features()
        btc.detect_market_regimes()
    else:
        if btc.data is None:
            logging.warning("Data has not been fetched in this session; make sure to load data before training or predicting.")

    if args.train:
        if btc.data is None:
            parser.error("No data available to train on. Use --fetch first or load data manually.")
        btc.train_all_models(test_size=0.1, lookback=args.lookback)

    if args.predict is not None:
        if btc.data is None or btc.xgb_model is None:
            parser.error("Models or data not available. Run --train first or supply trained models.")
        btc.predict_with_cycle_awareness(lookback=args.lookback, days_ahead=args.predict)

    if not any([args.fetch, args.train, args.predict is not None]):
        parser.print_help()


if __name__ == "__main__":
    main()