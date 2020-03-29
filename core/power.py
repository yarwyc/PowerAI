# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import os
import numpy as np
import pandas as pd
import datetime
from io import StringIO
from itertools import chain
from collections import Counter

from common.time_util import timer
from core.power_def import format_key, format_type, file_format, index_dict, \
    output_format, multi_line_header, useless_column, \
    set_format_version, get_format_version
from core.topo import PowerGraph


def f_name_conv(x):
    """
    Trim string, used in reading LFs.

    :param x: str.
    :return: str.
    """
    return x.strip(' \'')


def f_vbase_conv(v):
    """
    Convert voltage level, from float to integer.

    :param v: float. Voltage base.
    :return: integer. Voltage level.
    """
    if v > 900.:
        return 1000
    elif v > 700.:
        return 750
    elif v > 450.:
        return 500
    elif v > 300.:
        return 330
    elif v > 200.:
        return 220
    elif v > 100.:
        return 110
    elif v > 30.:
        return 35
    elif v > 5.:
        return 20
    return 1


def load_elem_info(file_name, etypes=[], index_col=['name']):
    """
    Load element infos.

    :param file_name: str.
    :param etypes: [str]. Used element types.
    :param index_col: list of str. Index.
    :return: pd.DataFrame.
    """
    elems = pd.read_table(file_name, encoding='gbk', sep=' ', index_col=index_col)
    if len(etypes) > 0:
        elems = elems[elems['type'].isin(etypes)]
    return elems


def load_station_info(file_name, index_col=['name']):
    """
    Load station infos.

    :param file_name: str.
    :param index_col: list of str. Index.
    :return: pd.DataFrame.
    """
    st_info = pd.read_table(file_name, encoding='gbk', sep=' ', index_col=index_col)
    return st_info


class Power:
    """
    Power class deal with PSASP data.

    Attributes:
        format_key, format_type, file_format, index_dict, output_format,
        multi_line_header, useless_columns:
            dict. Configuration for loading and saving LFs.
        fmt: str. Format template, 'on', 'off' or user defined.
        data: dict of pd.DataFrame. All loaded data are stored here.
        station: pd.DataFrame. Station infos.
    """

    def __init__(self, fmt):
        self.format_key = format_key()
        self.format_type = format_type()
        self.file_format = file_format()
        self.index_dict = index_dict()
        self.output_format = output_format()
        self.multi_line_header = multi_line_header()
        self.useless_columns = useless_column()

        self.fmt = fmt
        self.data = {}
        self.stations = None

    def get_column_and_index(self, file_name, fmt='on', ex=None):
        """
        Get relevent infos of the file, including etype, columns, index and ex.

        :param file_name: std.
        :param fmt: str. Format template.
        :param ex: str. None for getting extension name from file_name.
        :return: (etype, columns, index, ex).
        """
        if ex is None and '.' in file_name:
            ex = file_name[file_name.rindex('.') + 1:].lower()
        if fmt not in self.file_format or ex not in self.format_key:
            raise NotImplementedError("Not recognized format: (%s, %s)" % (fmt, ex))
        if ex in self.format_type['single']:
            return self.format_key[ex], self.file_format[fmt][ex], \
                   self.index_dict.get(ex, None), ex
        etypes = self.format_key[ex]
        columns = [self.file_format[fmt].get(ex + '_' + t, []) for t in etypes]
        indices = [self.index_dict.get(t, None) for t in etypes]
        return etypes, columns, indices, ex

    def get_flat_columns(self, fmt, etype, ex):
        """
        Get flatten columns.

        :param fmt: str.
        :param etype: str.
        :param ex: str.
        :return: [columns].
        """
        if ex in self.format_type['single']:
            return self.file_format[fmt][ex]
        else:
            return list(chain(*self.file_format[fmt][ex + '_' + etype]))

    def get_flag_valid(self, df, ex):
        """
        Get valid lp or st lines.

        :param df: pd.DataFrame.
        :param ex: str.
        :return: pd.Series of bool.
        """
        if ex in self.format_key['lp']:
            return Power.get_flag(df, 'lp')
        elif ex in self.format_key['st']:
            return Power.get_flag(df, 'st')
        else:
            return pd.Series(data=True, index=df.index)

    def load_lf(self, file_name, fmt='on', ex=None, drop_useless=True):
        """
        Load LF.L* files.

        :param file_name: str.
        :param fmt: str.
        :param ex: str.
        :param drop_useless: bool. Drop useless columns to save memory.
                                   It would use default value for saving file.
        :return: pd.DataFrame.
        """
        etype, columns, indices, ex = self.get_column_and_index(file_name, fmt, ex)
        if ex == 'l1':
            converters = {'name': f_name_conv, 'st_name': f_name_conv}
        else:
            converters = {'name': f_name_conv} if fmt == 'on' \
                else {'name': f_name_conv, 'name_ctrl': f_name_conv}
        if drop_useless and etype in self.useless_columns:
            usecols = [col for col in columns if col not in self.useless_columns[etype]]
        else:
            usecols = columns
        df = pd.read_csv(file_name, names=columns, encoding='gbk', index_col=False,
                         usecols=usecols, converters=converters)

        if ex == 'l1':
            df.index = range(1, df.shape[0] + 1)
            df['bus'] = range(1, df.shape[0] + 1)
        else:
            df['ori_order'] = range(df.shape[0])
            if ex == 'l3':
                df['ibus'] = np.abs(df['ibus'])
            if indices is not None:
                df.set_index(indices, drop=False, inplace=True)
                df.sort_index(inplace=True)
        return df

    def save_lf(self, file_name, df, fmt='on', ex=None, ori_order=True, miss='fill'):
        """
        Save LF.L* files.

        :param file_name: str.
        :param df: pd.DataFrame.
        :param fmt: str.
        :param ex: str.
        :param ori_order: bool. Use the original order while the df is loaded.
        :param miss: str, fill or raise. Fill default value or raise exception.
        """
        etype, columns, _, ex = self.get_column_and_index(file_name, fmt, ex)
        if ori_order and 'ori_order' in df.columns:
            columns = columns + ['ori_order']
        miss_columns = [col for col in columns if col not in df.columns]
        ready_columns = [col for col in columns if col in df.columns]
        if miss_columns and miss == 'raise':
            raise ValueError('Save miss columns: ', miss_columns)
        valid = self.get_flag_valid(df, ex)
        sub = df.loc[valid, ready_columns].copy()
        if len(sub) > 0:
            if 'ori_order' in sub.columns:
                sub.sort_values(by='ori_order', inplace=True)
                sub.drop(columns=['ori_order'], inplace=True)
                columns.remove('ori_order')
            if miss_columns and miss == 'fill':
                self.fill_default(sub, ex, miss_columns)
                sub = sub[columns]
            if ex == 'l3':
                t3w = sub['trs_type'].isin([31, 32, 33])
                sub.loc[t3w, 'ibus'] = -sub.loc[t3w, 'ibus']
        header = '' if ex != 'lp1' else '1,' + str(datetime.datetime.now())
        formats = self.get_output_format(ex, columns=sub.columns, dtypes=sub.dtypes)
        np.savetxt(file_name, sub.values, fmt=formats,
                   newline=',\n', header=header, comments=' ', encoding='gbk')

    def load_lp(self, file_name, base=None, fmt='on', ex=None, flag='lp',
                drop_useless=True):
        """
        Load LF.LP* and ST.S* files.

        :param file_name: str.
        :param base: pd.DataFrame. Base df of LF.
        :param fmt: str.
        :param ex: str.
        :param flag: str. lp or st.
        :param drop_useless: bool.
        :return: pd.DataFrame. Concatenated df.
        """
        etype, columns, indices, ex = self.get_column_and_index(file_name, fmt, ex)
        skiprows = 1 if ex == 'lp1' else 0
        if drop_useless and etype in self.useless_columns:
            usecols = [col for col in columns if col not in self.useless_columns[etype]]
        else:
            usecols = columns
        df = pd.read_csv(file_name, names=columns, encoding='gbk',
                         skiprows=skiprows, usecols=usecols, index_col=False)
        df.set_index(indices, inplace=True)
        Power.set_flag(df, True, flag)
        if base is None:
            return df
        df = df.reindex(index=base.index, fill_value=0)
        Power.set_flag(base, df['flag'] > 0, flag)
        columns = [col for col in df.columns if col not in base.columns]
        return pd.concat([base, df[columns]], axis=1)

    def load_st(self, file_name, base=None, fmt='on', ex=None, flag='st',
                drop_useless=True):
        """
        Load ST.S* files using load_lp with the same parameter.

        :return: pd.DataFrame. Concatenated df.
        """
        if ex is None and '.' in file_name:
            ex = file_name[file_name.rindex('.') + 1:].lower()
        if ex == 's1':
            if base is not None:
                Power.set_flag(base, True, flag)
            return None
        return self.load_lp(file_name, base=base, fmt=fmt, ex=ex, flag=flag,
                            drop_useless=drop_useless)

    def load_mlf(self, file_or_buffer, base=None, fmt='on',
                 ex=None, header_char=None, drop_useless=True):
        """
        Load multi-line files, including NL4, NP4, NS4, ML4, MP4.

        :param file_or_buffer: str or StringIO.
        :param base: pd.DataFrame. Base df of LF.
        :param fmt: str.
        :param ex: str.
        :param header_char: Separator of header, None for NL4, NP4, NS4, # for ML4, MP4.
        :param drop_useless: str.
        :return: {str: pd.DataFrame}. {etype: df}.
        """
        if isinstance(file_or_buffer, StringIO):
            source = file_or_buffer
            assert ex is not None
        elif isinstance(file_or_buffer, str):
            source = open(file_or_buffer, 'r', encoding='gbk')
        else:
            raise NotImplementedError("Unknown format. " + type(file_or_buffer))
        names, columns, indices, ex = self.get_column_and_index(file_or_buffer, fmt, ex)
        flag = 'lf'
        if ex in self.format_key['lp']:
            flag = 'lp'
        elif ex in self.format_key['st']:
            flag = 'st'
        num_lines = [len(col) for col in columns]
        columns = [list(chain(*col)) for col in columns]
        tables = dict(zip(names, [None] * len(names)))
        idx, i = -1, 0
        prompt = ''
        buffer = StringIO()
        lines = []
        for line in source:
            if header_char is not None and line[0] == header_char:
                # curr = line[:line.index(',')]
                curr = line[:2]     #sometimes lack of ','
                if prompt == curr:
                    continue
                if buffer.tell() > 0 and num_lines[idx] > 0:
                    buffer.seek(0)
                    if drop_useless and names[idx] in self.useless_columns:
                        usecols = [col for col in columns[idx]
                                   if col not in self.useless_columns[names[idx]]]
                    else:
                        usecols = columns[idx]
                    df = pd.read_csv(buffer, names=columns[idx], usecols=usecols,
                                     converters={'name': f_name_conv})
                    if indices[idx] is not None:
                        df.set_index(indices[idx], drop=False, inplace=True)
                        # df.sort_index(inplace=True)
                    Power.set_flag(df, True, flag)
                    tables[names[idx]] = df
                buffer.truncate(0)
                buffer.seek(0)
                lines.clear()
                prompt = curr
                i = 0
                idx += 1
            elif num_lines[idx] > 0:
                lines.append(line.rstrip('\r\n'))
                i += 1
                if i % num_lines[idx] == 0:
                    buffer.write(''.join(lines) + '\n')
                    lines.clear()
        if isinstance(file_or_buffer, str):
            source.close()
        if buffer.tell() > 0 and idx < len(names):
            idx = max(idx, 0)
            buffer.seek(0)
            if drop_useless and names[idx] in self.useless_columns:
                usecols = [col for col in columns[idx]
                           if col not in self.useless_columns[names[idx]]]
            else:
                usecols = columns[idx]
            df = pd.read_csv(buffer, names=columns[idx], usecols=usecols,
                             converters={'name': f_name_conv})
            if indices[idx] is not None:
                df.set_index(indices[idx], drop=False, inplace=True)
                # df.sort_index(inplace=True)
            Power.set_flag(df, True, flag)
            tables[names[idx]] = df
        buffer.close()
        if base is None:
            return tables
        for k in tables:
            if tables[k] is None or len(tables[k]) == 0:
                continue
            if k in base and base[k] is not None:
                tables[k] = tables[k].reindex(index=base[k].index, fill_value=0)
                Power.set_flag(base[k], tables[k]['flag'] > 0, flag)
                columns = [col for col in tables[k].columns if col not in base[k].columns]
                base[k] = pd.concat([base[k], tables[k][columns]], axis=1)
            else:
                base[k] = tables[k]
        return base

    def save_mlf(self, file_name, data, fmt='on', ex=None, miss='fill'):
        """
        Save multi-line files, including NL4, NP4, NS4, ML4, MP4.

        :param file_name: str.
        :param data: {str: pd.DataFrame}. {etype: df}.
        :param fmt: str.
        :param ex: str.
        :param miss: str, fill or raise. Fill default value or raise exception.
        """
        etypes, columns, _, ex = self.get_column_and_index(file_name, fmt, ex)
        headers = self.multi_line_header[ex]
        with open(file_name, 'w') as fp:
            for i, e in enumerate(etypes):
                if headers is not None and i < len(headers):
                    fp.write('\n'.join(headers[i]) + '\n')
                if not columns[i] or e not in data \
                        or data[e] is None or len(data[e]) == 0:
                    print('Not recognized format (%s, %s) or empty data' % (ex, e))
                    continue
                cols = list(chain(*columns[i]))
                miss_cols = [col for col in cols if col not in data[e].columns]
                ready_cols = [col for col in cols if col in data[e].columns]
                if miss_cols and miss == 'raise':
                    raise ValueError('Save miss columns: ', miss_cols)
                valid = self.get_flag_valid(data[e], ex)
                sub = data[e].loc[valid, ready_cols].copy()
                if miss_cols and miss == 'fill' and len(sub) > 0:
                    self.fill_default(sub, ex, columns=miss_cols)
                    sub = sub[cols]
                out_fmt = [self.get_output_format(ex, columns=col, dtypes=sub.dtypes[col])
                           for col in columns[i]]
                out_fmt = ',\n'.join(out_fmt) + ',\n'
                for _, *values in sub.itertuples():
                    fp.write(out_fmt % tuple(values))

    def get_output_format(self, ex, columns, dtypes):
        """
        Get output format string.

        :param ex: str.
        :param columns: [str].
        :param dtypes: pd.Series.
        :return: str.
        """
        default = self.output_format['default']
        if ex not in self.output_format:
            formats = [default[dtypes[col].kind][1] for col in columns]
        else:
            formats = [self.output_format[ex].get(col, default[dtypes[col].kind])[1]
                       for col in columns]
        return ','.join(formats)

    def check_columns(self, df, etype, columns=None, ex=None, fmt='on'):
        """
        Get the missing columns of designated file type.

        :param df: pd.DataFrame.
        :param etype: str. Equipment type.
        :param columns: [str]. Target columns.
        :param ex: str. Extension name of file.
        :param fmt: str. Format template.
        :return: [str]. Missing columns.
        """
        if not columns:
            if isinstance(self.format_key[ex], list):
                columns = list(chain(self.file_format[fmt][ex + '_' + etype]))
            else:
                columns = self.file_format[fmt][ex]
        return [col for col in df.columns if col not in columns]

    def fill_default(self, df, ex, columns):
        """
        Fill default values of designated columns.

        :param df: pd.DataFrame.
        :param ex: str. Extension name.
        :param columns: [str]. Target columns.
        """
        for col in columns:
            value = self.output_format[ex][col][0]
            if value is None:
                raise ValueError('Column [%s] has no default value' % col)
            if col not in df.columns:
                df[col] = value
            else:
                df[col].fillna(value, inplace=True)

    @staticmethod
    def set_flag(df, valid, flag):
        """
        Set each line's flag of lp or st.
        :param df: pd.DataFrame.
        :param valid: bool or [bool].
        :param flag: str, lp or st.
        """
        if flag == 'lp':
            bit = 1
        elif flag == 'st':
            bit = 2
        else:
            return
        bits = valid * bit
        if 'flag' not in df.columns:
            df['flag'] = bits
        else:
            df['flag'] &= ~bit
            df['flag'] |= bits

    @staticmethod
    def get_flag(df, flag):
        """
        Get each line's flag of lp or st.
        :param df: pd.DataFrame.
        :param flag: str, lp or st.
        :return: pd.Series. Validation of each line.
        """
        if 'flag' not in df.columns:
            return pd.Series(data=False, index=df.index, name='flag')
        bit = 1 if flag == 'lp' else 2
        return df['flag'] & bit > 0

    def drop_data(self, etype, fmt, flag):
        """
        Drop lp or st data.

        :param etype: str. Equipment type.
        :param fmt: str.
        :param flag: str, lp or st.
        """
        if flag == 'lp':
            ex1 = self.format_key[etype][1]
        elif flag == 'st':
            ex1 = self.format_key[etype][2]
        flags = Power.get_flag(self.data[etype], flag)
        if ex1 is None or not np.any(flags):
            return
        columns1 = self.get_flat_columns(fmt, etype, ex1)
        columns0 = self.get_flat_columns(fmt, etype, self.format_key[etype][0])
        drops = [col for col in columns1 if col not in columns0]
        self.data[etype].drop(columns=drops, inplace=True)
        Power.set_flag(self.data[etype], False, flag)

    def describe(self):
        """
        Describe lf, lp, st record amounts for each etype.
        """
        for name, df in self.data.items():
            n_lf = df.shape[0]
            n_lp = Power.get_flag(df, 'lp').sum()
            n_st = Power.get_flag(df, 'st').sum()
            print('[%s]: n_lf=%d, n_lp=%d, n_st=%d' % (name, n_lf, n_lp, n_st))

    def load_power(self, path, fmt='on', lp=True, st=True, station=True, shorten=True):
        """
        Load power from data directory.

        :param path: str.
        :param fmt: str.
        :param lp: bool. Whether load lp files.
        :param st: bool. Whether load st files.
        :param station: bool. Whether load station infos.
        :param shorten: bool. Use shorten storage.
        """
        self.data['bus'] = self.load_lf(path + '/LF.L1', fmt)
        self.data['acline'] = self.load_lf(path + '/LF.L2', fmt)
        self.data['transformer'] = self.load_lf(path + '/LF.L3', fmt)
        self.data['generator'] = self.load_lf(path + '/LF.L5', fmt)
        self.data['load'] = self.load_lf(path + '/LF.L6', fmt)
        self.data.update(self.load_mlf(path + '/LF.NL4', fmt=fmt))
        self.data.update(self.load_mlf(path + '/LF.ML4', fmt=fmt, header_char='#'))
        self.data['dcbus']['name'] = self.data['bus'].loc[self.data['dcbus']['bus'],
                                                          'name']
        if lp:
            self.data['bus'] = self.load_lp(path + '/LF.LP1', self.data['bus'], fmt)
            self.data['acline'] = self.load_lp(path + '/LF.LP2',
                                               self.data['acline'], fmt)
            self.data['transformer'] = self.load_lp(path + '/LF.LP3',
                                                    self.data['transformer'], fmt)
            self.data['generator'] = self.load_lp(path + '/LF.LP5',
                                                  self.data['generator'], fmt)
            self.data['load'] = self.load_lp(path + '/LF.LP6', self.data['load'], fmt)
            self.data.update(self.load_mlf(path + '/LF.NP4', self.data, fmt))
            self.data.update(self.load_mlf(path + '/LF.MP4', self.data, fmt,
                                           header_char='#'))
        if st:
            # self.data['bus'] = self.load_st(path+'/ST.S1', self.data['bus'], fmt)
            Power.set_flag(self.data['bus'], True, 'st')
            self.data['acline'] = self.load_st(path + '/ST.S2', self.data['acline'], fmt)
            self.data['transformer'] = self.load_st(path + '/ST.S3',
                                                    self.data['transformer'], fmt)
            self.data['generator'] = self.load_st(path + '/ST.S5',
                                                  self.data['generator'], fmt)
            self.data['load'] = self.load_st(path + '/ST.S6', self.data['load'], fmt)
            self.data.update(self.load_mlf(path + '/ST.NS4', self.data, fmt))
            self.data['vsc'] = self.load_st(path + '/ST.MS4', self.data['vsc'], fmt)
        if shorten:
            self.shorten_storage(fmt=fmt)
        if station:
            self.generate_station_info()
        # self.describe()

    def shorten_storage(self, etypes=None, fmt='on'):
        """
        Shorten the storage format, int32 for integer, float32 for float.

        :param etypes: str.
        :param fmt: str.
        """
        if not etypes:
            etypes = self.data.keys()
        else:
            etypes = [e for e in etypes if e in self.data]
        for e in etypes:
            if not isinstance(self.data[e], pd.DataFrame):
                continue
            dtypes = self.data[e].dtypes
            ex = self.format_key[e][1]
            lp_columns = self.get_flat_columns(fmt, e, ex)
            columns = [col for col in self.data[e].columns if dtypes[col].kind == 'i']
            if columns:
                self.data[e][columns] = self.data[e][columns].astype('int32')
            columns = [col for col in self.data[e].columns
                       if dtypes[col].kind == 'f' and col not in lp_columns]
            if columns:
                self.data[e][columns] = self.data[e][columns].astype('float32')

    def generate_mdc_version_outline(self, fmt='on'):
        """
        Generate mdc version and outline infos.

        :param fmt: str.
        """
        self.data['version'] = pd.DataFrame([get_format_version('mdc')],
                                            columns=['version'])
        Power.set_flag(self.data['version'], True, 'lp')
        columns = self.file_format[fmt]['ml4_outline'] \
                  + self.file_format[fmt]['mp4_outline']
        columns = list(chain(*columns))
        values = [0] * 12
        values[5] = 1  # method
        if 'dcbus' in self.data and self.data['dcbus'] is not None:
            values[0] = self.data['dcbus'].shape[0]
            values[7] = np.sum(Power.get_flag(self.data['dcbus'], 'lp'))
        if 'vsc' in self.data and self.data['vsc'] is not None:
            values[1] = self.data['vsc'].shape[0]
            values[8] = np.sum(Power.get_flag(self.data['vsc'], 'lp'))
        if 'lcc' in self.data and self.data['lcc'] is not None:
            values[2] = self.data['lcc'].shape[0]
            values[9] = np.sum(Power.get_flag(self.data['lcc'], 'lp'))
        if 'mdcline' in self.data and self.data['mdcline'] is not None:
            values[3] = self.data['mdcline'].shape[0]
            values[10] = np.sum(Power.get_flag(self.data['mdcline'], 'lp'))
        if 'dcdc' in self.data and self.data['dcdc'] is not None:
            values[4] = self.data['dcdc'].shape[0]
            values[11] = np.sum(Power.get_flag(self.data['dcdc'], 'lp'))
        if 'dcsys' in self.data and self.data['dcsys'] is not None:
            values[6] = self.data['dcsys'].shape[0]
        self.data['outline'] = pd.DataFrame([values], columns=columns, index=[0])
        Power.set_flag(self.data['outline'], True, 'lp')

    def generate_island_info(self):
        """
        Generate island info.
        """
        graph = PowerGraph(self, graph_type='multi', node_type='bus')
        islands = graph.get_islands(10)
        self.data['bus']['island'] = islands

    def get_largest_island(self):
        """
        Get the largest island.
        :return: int. No. of the largest island.
        """
        if 'island' not in self.data['bus'].columns:
            self.generate_island_info()
        counts = self.data['bus']['island'].value_counts()
        if counts.shape[0] == 0:
            raise ValueError("No island data.")
        return counts.index[0]

    def generate_station_info(self):
        """
        Generate station info.
        """
        if 'island' not in self.data['bus'].columns:
            self.generate_island_info()
        names = ['%s_%d' % (n, i) for idx, n, i in
                 self.data['bus'][['st_name', 'island']].itertuples()]
        self.stations = pd.DataFrame(data=list(set(names)), columns=['name'])
        self.stations['island'] = \
            [int(name.split('_')[-1]) for name in self.stations['name']]
        self.stations['ori_name'] = \
            [name[:name.rindex('_')] for name in self.stations['name']]
        name_idx = pd.DataFrame(data=range(self.stations.shape[0]),
                                index=self.stations.name)
        self.data['bus']['st_no'] = name_idx.loc[names].values
        self.data['acline']['st_i'] = \
            self.data['bus'].loc[self.data['acline'].ibus, 'st_no'].values
        self.data['acline']['st_j'] = \
            self.data['bus'].loc[self.data['acline'].jbus, 'st_no'].values
        self.data['transformer']['st_no'] = \
            self.data['bus'].loc[self.data['transformer'].ibus, 'st_no'].values
        self.data['generator']['st_no'] = \
            self.data['bus'].loc[self.data['generator'].bus, 'st_no'].values
        self.data['load']['st_no'] = \
            self.data['bus'].loc[self.data['load'].bus, 'st_no'].values

    def save_power(self, path, fmt='on', lp=True, st=True, miss='fill'):
        """
        Save power to power directory.

        :param path: str.
        :param fmt: str.
        :param lp: bool. Whether save lp files.
        :param st: bool. Whether save st files.
        :param miss: str, fill or raise.
        """
        if fmt != self.fmt:
            print("Format mismatch: load is %s, save is %s" % (self.fmt, fmt))
        if not os.path.exists(path):
            os.mkdir(path)
        self.save_lf(path + '/LF.L1', self.data['bus'], fmt=fmt,
                     ori_order=False, miss=miss)
        self.save_lf(path + '/LF.L2', self.data['acline'], fmt=fmt, miss=miss)
        self.save_lf(path + '/LF.L3', self.data['transformer'], fmt=fmt, miss=miss)
        self.save_lf(path + '/LF.L5', self.data['generator'], fmt=fmt, miss=miss)
        self.save_lf(path + '/LF.L6', self.data['load'], fmt=fmt, miss=miss)
        self.save_mlf(path + '/LF.NL4', self.data, fmt=fmt, miss=miss)
        self.save_mlf(path + '/LF.ML4', self.data, fmt=fmt, miss=miss)
        if lp:
            self.save_lf(path + '/LF.LP1', self.data['bus'], fmt=fmt,
                         ori_order=False, miss=miss)
            self.save_lf(path + '/LF.LP2', self.data['acline'], fmt=fmt, miss=miss)
            self.save_lf(path + '/LF.LP3', self.data['transformer'], fmt=fmt, miss=miss)
            self.save_lf(path + '/LF.LP5', self.data['generator'], fmt=fmt, miss=miss)
            self.save_lf(path + '/LF.LP6', self.data['load'], fmt=fmt, miss=miss)
            self.save_mlf(path + '/LF.NP4', self.data, fmt=fmt, miss=miss)
            self.save_mlf(path + '/LF.MP4', self.data, fmt=fmt, miss=miss)
        if st:
            self.save_lf(path + '/ST.S1', self.data['bus'], fmt=fmt,
                         ori_order=False, miss=miss)
            self.save_lf(path + '/ST.S2', self.data['acline'], fmt=fmt, miss=miss)
            self.save_lf(path + '/ST.S3', self.data['transformer'], fmt=fmt, miss=miss)
            self.save_lf(path + '/ST.S5', self.data['generator'], fmt=fmt, miss=miss)
            self.save_lf(path + '/ST.S6', self.data['load'], fmt=fmt, miss=miss)
            self.save_mlf(path + '/ST.NS4', self.data, fmt=fmt, miss=miss)
            self.save_lf(path + '/ST.MS4', self.data['vsc'], fmt=fmt, miss=miss)

    @staticmethod
    def statistic_acline(aclines):
        """
        Statistic acline.

        :param aclines: pd.DataFrame.
        """
        valid_on = aclines['mark'] > 0
        valid_pc = aclines['ibus'] == aclines['jbus']
        # valid_r0 = aclines['r'] == 0.
        valid_x0 = aclines['x'].abs() <= 1e-4
        valid_xn = aclines['x'] < -1e-4
        valid_xp = aclines['x'] > 1e-4
        count_ac = aclines.shape[0]
        count_ac_on = np.sum(valid_on)
        count_pc = np.sum(valid_pc & valid_xn)
        count_pc_on = np.sum(valid_pc & valid_xn & valid_on)
        count_pr = np.sum(valid_pc & valid_xp)
        count_pr_on = np.sum(valid_pc & valid_xp & valid_on)
        count_sr = np.sum(~valid_pc & valid_xn)
        count_sr_on = np.sum(~valid_pc & valid_xn & valid_on)
        count_bs = np.sum(valid_x0)
        count_bs_on = np.sum(valid_x0 & valid_on)
        print('acline=%d, on=%d' % (count_ac, count_ac_on))
        print('pc=%d, on=%d' % (count_pc, count_pc_on))
        print('pr=%d, on=%d' % (count_pr, count_pr_on))
        print('sr=%d, on=%d' % (count_sr, count_sr_on))
        print('bs=%d, on=%d' % (count_bs, count_bs_on))


if __name__ == '__main__':
    # set_format_version({'mdc': 2.3})
    path = 'D:/PSA_src/psa/localdata/0913/data'
    fmt = 'on'
    path = 'D:/PSASP_Pro/2020国调年度/冬低731'
    fmt = 'off'
    power = Power(fmt)
    with timer('Load power'):
        power.load_power(path, fmt=fmt, lp=False, st=True, station=False)
    with timer('Save power'):
        power.save_power(path + '/out', lp=False, st=True, fmt=fmt)
