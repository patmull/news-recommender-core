from celery import Celery
from urllib.parse import urlparse

#REDIS_URL = 'redis://default:RuTYjOqZZofckHAPTNqUMlg8XjT1nQdZ@redis-18878.c59.eu-west-1-2.ec2.cloud.redislabs.com:18878'
# REDIS_URL = 'amqp://localhost//'


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery