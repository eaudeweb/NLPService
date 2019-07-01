""" Various utilities
"""


def get_keys_by_prefix(settings, prefix):
    for k, v in settings.items():
        if not k.startswith(prefix):
            continue

        yield k, v
