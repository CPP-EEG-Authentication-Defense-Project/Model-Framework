import logging


LOGGER_NAME = 'eeg-auth-model-framework'


class PrefixedLoggingAdapter(logging.LoggerAdapter):
    """
    Simple logger adapter which allows for prefixing all log messages with static text.
    """
    def __init__(self, prefix: str, logger: logging.Logger, **kwargs):
        super().__init__(logger, kwargs)
        self.prefix = prefix

    def process(self, msg, kwargs):
        # override process method, add prefix to message.
        return '%s %s' % (self.prefix, msg), kwargs
