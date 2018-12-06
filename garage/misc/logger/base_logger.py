"""Base Logger."""
# flake8: noqa
from contextlib import contextmanager
import csv
import datetime
from enum import Enum
import json
import os
import os.path as osp
import sys

import dateutil.tz
import joblib
import numpy as np

from garage.misc.console import colorize, mkdir_p
from garage.misc.logger.tabulate import tabulate


class TerminalTablePrinter:
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum':
                o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        return json.JSONEncoder.default(self, o)


class Logger():
    def __init__(self):
        self._prefixes = []
        self._prefix_str = ''

        self._tabular_prefixes = []
        self._tabular_prefix_str = ''

        self._tabular = []

        self._text_outputs = []
        self._tabular_outputs = []

        self._text_fds = {}
        self._tabular_fds = {}
        self._tabular_header_written = set()

        self._snapshot_dir = None
        self._snapshot_mode = 'all'
        self._snapshot_gap = 1

        self._log_tabular_only = False
        self._header_printed = False

        self.table_printer = TerminalTablePrinter()

    def _add_output(self, file_name, arr, fds, mode='a'):
        if file_name not in arr:
            mkdir_p(os.path.dirname(file_name))
            arr.append(file_name)
            fds[file_name] = open(file_name, mode)

    def _remove_output(self, file_name, arr, fds):
        if file_name in arr:
            fds[file_name].close()
            del fds[file_name]
            arr.remove(file_name)

    def push_prefix(self, prefix):
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def add_text_output(self, file_name):
        self._add_output(
            file_name, self._text_outputs, self._text_fds, mode='a')

    def remove_text_output(self, file_name):
        self._remove_output(file_name, self._text_outputs, self._text_fds)

    def add_tabular_output(self, file_name):
        self._add_output(
            file_name, self._tabular_outputs, self._tabular_fds, mode='w')

    def remove_tabular_output(self, file_name):
        if self._tabular_fds[file_name] in self._tabular_header_written:
            self._tabular_header_written.remove(self._tabular_fds[file_name])
        self._remove_output(file_name, self._tabular_outputs,
                            self._tabular_fds)

    def set_snapshot_dir(self, dir_name):
        mkdir_p(dir_name)
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self):
        return self._snapshot_dir

    def get_snapshot_mode(self):
        return self._snapshot_mode

    def set_snapshot_mode(self, mode):
        self._snapshot_mode = mode

    def get_snapshot_gap(self):
        return self._snapshot_gap

    def set_snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def set_log_tabular_only(self, log_tabular_only):
        self._log_tabular_only = log_tabular_only

    def get_log_tabular_only(self):
        return self._log_tabular_only

    def log(self, s, with_prefix=True, with_timestamp=True, color=None):
        out = s
        if with_prefix:
            out = self._prefix_str + out
        # out_basic holds output with a simpler timestamp for stdout
        out_basic = out
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp_basic = now.strftime('%Y-%m-%d %H:%M:%S')
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out_basic = "%s | %s" % (timestamp_basic, out_basic)
            out = "%s | %s" % (timestamp, out)
        if color is not None:
            out = colorize(out, color)
            out_basic = colorize(out_basic, color)
        if not self._log_tabular_only:
            # Also log to stdout
            print(out_basic)
            for fd in list(self._text_fds.values()):
                fd.write(out + '\n')
                fd.flush()
            sys.stdout.flush()

    def push_tabular_prefix(self, key):
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def pop_tabular_prefix(self):
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    @contextmanager
    def prefix(self, key):
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    @contextmanager
    def tabular_prefix(self, key):
        self.push_tabular_prefix(key)
        yield
        self.pop_tabular_prefix()

    def record_tabular(self, key, val):
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))

    def dump_tabular(self, *args, **kwargs):
        wh = kwargs.pop("write_header", None)
        if self._tabular:
            if self._log_tabular_only:
                self.table_printer.print_tabular(self._tabular)
            else:
                for line in tabulate(self._tabular).split('\n'):
                    self.log(line, *args, **kwargs)
            tabular_dict = dict(self._tabular)

            # Also write to the csv files
            # This assumes that the keys in each iteration won't change!
            for tabular_fd in list(self._tabular_fds.values()):
                writer = csv.DictWriter(
                    tabular_fd, fieldnames=list(tabular_dict.keys()))
                if wh or (wh is None
                          and tabular_fd not in self._tabular_header_written):
                    writer.writeheader()
                    self._tabular_header_written.add(tabular_fd)
                writer.writerow(tabular_dict)
                tabular_fd.flush()
            del self._tabular[:]

    def pop_prefix(self):
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)

    def save_itr_params(self, itr, params):
        if self._snapshot_dir:
            if self._snapshot_mode == 'all':
                file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                joblib.dump(params, file_name, compress=3)
            elif self._snapshot_mode == 'last':
                # override previous params
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                joblib.dump(params, file_name, compress=3)
            elif self._snapshot_mode == "gap":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir,
                                         'itr_%d.pkl' % itr)
                    joblib.dump(params, file_name, compress=3)
            elif self._snapshot_mode == 'none':
                pass
            else:
                raise NotImplementedError

    def log_parameters_lite(self, log_file, args):
        log_params = {}
        for param_name, param_value in args.__dict__.items():
            log_params[param_name] = param_value
        if args.args_data is not None:
            log_params["json_args"] = dict()
        mkdir_p(os.path.dirname(log_file))
        with open(log_file, "w") as f:
            json.dump(log_params, f, indent=2, sort_keys=True, cls=MyEncoder)

    def log_variant(self, log_file, variant_data):
        mkdir_p(os.path.dirname(log_file))
        if hasattr(variant_data, "dump"):
            variant_data = variant_data.dump()
        with open(log_file, "w") as f:
            json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)

    def record_tabular_misc_stat(self, key, values, placement='back'):
        if placement == 'front':
            prefix = ""
            suffix = key
        else:
            prefix = key
            suffix = ""
        if values:
            self.record_tabular(prefix + "Average" + suffix,
                                np.average(values))
            self.record_tabular(prefix + "Std" + suffix, np.std(values))
            self.record_tabular(prefix + "Median" + suffix, np.median(values))
            self.record_tabular(prefix + "Min" + suffix, np.min(values))
            self.record_tabular(prefix + "Max" + suffix, np.max(values))
        else:
            self.record_tabular(prefix + "Average" + suffix, np.nan)
            self.record_tabular(prefix + "Std" + suffix, np.nan)
            self.record_tabular(prefix + "Median" + suffix, np.nan)
            self.record_tabular(prefix + "Min" + suffix, np.nan)
            self.record_tabular(prefix + "Max" + suffix, np.nan)

    def reset(self):
        pass
