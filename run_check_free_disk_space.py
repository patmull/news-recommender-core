import logging
import math
import os
import re
import shutil
import time
import schedule

import mail_sender

ALERT_LIMIT_MB = 200

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from " + os.path.basename(__file__))


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def disk_space_is_over_limit(free):
    free_space = convert_size(free)
    if 'MB' in free_space:
        size = re.sub('[^\d|\.]', '', free_space)
        logging.warning("Free disk space exceeded.")
        logging.warning(free_space)
        if ALERT_LIMIT_MB < float(size):
            return False
        else:
            return True


def check_free_space_job():
    total, used, free = shutil.disk_usage("/")
    if disk_space_is_over_limit(free):
        logging.info("Sending e-mail.")
        mail_sender.send_error_email("FREE DISK SPACE ON EXCEEDED %s. Please react immediately and free disk space,"
                                     "some functions of the system may not continue otherwise." % ALERT_LIMIT_MB)


def main():
    check_free_space_job()  # for testing purposes (immidiately triggers the method)

    schedule.every(30).minutes.do(check_free_space_job)

    while 1:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__": main()
