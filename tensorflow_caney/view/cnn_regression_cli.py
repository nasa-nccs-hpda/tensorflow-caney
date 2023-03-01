import sys
import time
import logging
import argparse
from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression


# -----------------------------------------------------------------------------
# main
#
# python dsm_pipeline_cli.py -c config.yaml -d config.csv -s preprocess train
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to perform CNN regression.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument('-d',
                        '--data-csv',
                        type=str,
                        required=True,
                        dest='data_csv',
                        help='Path to the data configuration file')

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

    # Setup timer to monitor script execution time
    timer = time.time()

    # Initialize pipeline object
    pipeline = CNNRegression(args.config_file, args.data_csv)

    # Regression CHM pipeline steps
    if "preprocess" in args.pipeline_step:
        pipeline.preprocess()
    if "train" in args.pipeline_step:
        pipeline.train()
    if "predict" in args.pipeline_step:
        pipeline.predict()

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
