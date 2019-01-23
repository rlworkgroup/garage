def get_now_timestamp():
    import datetime
    return datetime.datetime.now().isoformat()


def get_timestamp(name):
    import re
    isoformat_regex = r'\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d\.\d+'
    return re.search(isoformat_regex, name).group(0)


def cat_for_fname(*args):
    return '_'.join(args) + '.pkl'
