import logging

from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from try_hybrid_methods.")


def main():
    run_prefilling_collaborative(methods=["hybrid"], user_id=3124)


if __name__ == "__main__": main()
