from rabbitmq_publisher import notify_prefiller

@PendingDeprecationWarning
def invoke_notification():
    notify_prefiller()


invoke_notification()
