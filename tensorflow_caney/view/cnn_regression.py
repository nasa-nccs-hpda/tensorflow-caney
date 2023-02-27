import sys
import time
import logging
import argparse
from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:

    # Process command-line args.
    desc = 'CNN Regression Pipeline for Satellite Imagery'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-filename',
                        type=str,
                        required=True,
                        dest='config_filename',
                        help='Path to the configuration file')

    parser.add_argument('-d',
                        '--data-csv',
                        type=str,
                        required=False,
                        default=None,
                        dest='data_csv',
                        help='Path to the data CSV configuration file')

    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=['preprocess', 'train', 'predict'],
                        choices=['preprocess', 'train', 'predict'])

    args = parser.parse_args()

    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Initialize CNN Regression Pipeline
    cnn_pipeline = CNNRegression(
        args.config_filename, args.data_csv, logger)

    timer = time.time()

    # Regression CHM pipeline steps
    if "preprocess" in args.pipeline_step:
        cnn_pipeline.preprocess()
    #if "train" in args.pipeline_step:
    #    run_train(args, conf)
    #if "predict" in args.pipeline_step:
    #    run_predict(args, conf)

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
