import math
import re
import shutil
import time
import schedule

import mail_sender

ALERT_LIMIT_MB = 200


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def disk_space_is_over_limit(free):
    print(free)
    free_space = convert_size(free)
    print(free_space)
    if 'MB' in free_space:
        size = re.sub('[^\d|\.]', '', free_space)
        print("After conversion:")
        print(size)
        if ALERT_LIMIT_MB < float(size):
            return False
        else:
            return True


def check_free_space_job():
    total, used, free = shutil.disk_usage("/")
    if disk_space_is_over_limit(free):
        mail_sender.send_error_email("FREE DISK SPACE ON EXCEEDED %s. Please react immediately and free disk space,"
                                     "some functions of the system may not continue otherwise." % ALERT_LIMIT_MB)


def main():
    check_free_space_job()  # for testing purposes (immidiately triggers the method)

    schedule.every(5).minutes.do(check_free_space_job)

    while 1:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__": main()
