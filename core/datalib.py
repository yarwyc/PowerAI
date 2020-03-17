# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import pandas as pd
import numpy as np

from common.time_util import timer
from core.datalib_def import datalib_format

datalib_columns = datalib_format()


def get_datalib_info(file_name):
    """
    Get table name, start_line_num, end_line_num and description.

    :param file_name: str.
    :return: version, {table_name: (start_line_num, end_line_num, desc)}
    """
    infos = {}
    with open(file_name, 'r') as fp:
        line = fp.readline()
        version = line[2:line.rindex(',')]
        line = fp.readline()
        # para_amount = [int(i) for i in line.split(',')][:-1]
        name = ''
        for i, line in enumerate(fp):
            if line.startswith('#'):
                if name != '':
                    end_line_num = i + 1
                    infos[name] = (start_line_num, end_line_num, desc)
                start_line_num = i + 2  # considering the first 2 lines in file head
                values = line.split(',')
                name = values[1].lower().strip() if len(values) > 1 else ''
                desc = values[2] if len(values) > 3 else ''
                infos[name] = (start_line_num, start_line_num, desc)
    if '' in infos:
        infos.pop('')
    return version, infos


def load_datalib(file_name, model_names=None):
    """
    Read DATALIB.DAT by repeat mode.

    :param file_name: str.
    :param model_names: iterable str. None for reading all models.
    :return: {name: DataFrame}
    """
    version, infos = get_datalib_info(file_name)
    models = {'version': version}
    if model_names is None:
        names = infos.keys()
    else:
        names = [i.lower() for i in model_names]
    for name in set(names):
        if name not in infos:
            print("Model [%s] not found in %s" % (name, file_name))
            continue
        start, end, desc = infos[name]
        if name in datalib_columns:
            columns = datalib_columns[name]
            converters = {'note': lambda x: x.strip(' \'')} if 'note' in columns \
                else None
        else:
            columns, converters = None, None
        df = pd.read_csv(file_name, encoding='gbk', names=columns, usecols=columns,
                         converters=converters, skiprows=start + 1, nrows=end - start)
        if 'par_no' in df.columns:
            df.set_index(['par_no'], drop=False, inplace=True)
        elif 0 in df.columns:
            df.set_index([0], drop=False, inplace=True)
            # for unknown format, use the first columns as index
        df.sort_index(inplace=True)
        models[name] = df
    return models


def get_model_name(model_type, model_no):
    model_type = model_type.lower()
    if model_type == "avr":
        if model_no in range(3, 11):
            return 'avr310' if 'avr310' in datalib_columns else None
        elif model_no in range(11, 13):
            return 'avr1112' if 'avr1112' in datalib_columns else None
    if model_type in datalib_columns:
        return model_type
    elif model_type + str(model_no) in datalib_columns:
        return model_type + str(model_no)
    return None


if __name__ == '__main__':
    from core.power import Power
    path = 'D:/PSASP_Pro/2020国调年度/冬低731'
    fmt = 'off'
    power = Power(fmt)
    power.load_power(path, fmt=fmt, lp=False, st=True, station=True)
    with timer('Load datalib'):
        datalib = load_datalib(path+'/DATALIB.DAT')
    iwant = [('gen', 3, 11),
             ('avr', 12, 18),
             ('gov', 7, 3),
             ('pss', 2, 7)]
    for t, m, p in iwant:
        name = get_model_name(t, m)
        if name is not None:
            print(datalib[name].loc[p])
