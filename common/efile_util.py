# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import os
import pandas as pd
from io import StringIO

from common.time_util import timer


def get_efile_info(file_name):
    """
    Get table name, start_line_num, end_line_num and column names.

    :param file_name: str.
    :return: {table_name: (start_line_num, end_line_num, [column_names])}
    """
    infos = {}
    with open(file_name, 'r') as fp:
        for i, line in enumerate(fp):
            if line.startswith('</'):
                end_line_num = i
                infos[table_name] = (start_line_num, end_line_num, column_names)
            elif line.startswith('<'):
                table_name = line[1:line.index(':')]
                start_line_num = i
                column_names = []
            elif line.startswith('@'):
                column_names = line.split()  # include '@' column, for '#' in the content.
    return infos


def read_efile(file_name, table_names=None, use_columns=None):
    """
    Read E file by repeat mode.

    :param file_name: str.
    :param table_names: iterable str. None for reading all tables.
    :param use_columns: dict. Use columns for tables.
    :return: {name: DataFrame}
    """
    tables = {}
    infos = get_efile_info(file_name)
    if table_names is None:
        table_names = infos.keys()
    for name in set(table_names):
        if name not in infos:
            print("Table [%s] not found in %s" % (name, file_name))
            continue
        start, end, columns = infos[name]
        if not use_columns or name not in use_columns:
            usecols = columns[1:]
        else:
            usecols = use_columns[name]
        df = pd.read_table(file_name, encoding='gbk', sep='\s+',
                           names=columns, usecols=usecols,
                           skiprows=start + 3, nrows=end - start - 3)
        tables[name] = df
    return tables


def read_efile_buffer(file_name, table_names=None):
    """
    Read E file by buffer mode.
    It seems repeat mode is more efficient than buffer mode?

    :param file_name: str.
    :param table_names: iterable str. None for reading all tables.
    :return: {name: DataFrame}
    """
    tables = {}
    name = ""
    valid = False
    buffer = StringIO()
    with open(file_name, 'r', encoding='gbk') as fp:
        for line in fp:
            if line[0] == '<':
                if line[1] == '/':
                    if valid:
                        buffer.seek(0)
                        df = pd.read_table(buffer, sep='\s+',
                                           names=column_names,
                                           usecols=column_names[1:])
                        tables[name] = df
                        valid = False
                else:
                    name = line[1:line.index(':')]
                    if table_names is None or name in table_names:
                        buffer.truncate(0)
                        valid = True
            elif line[0] == '@':
                column_names = line.split()  # include '@' column, for '#' in the content.
            elif line[0] == '#':
                if valid:
                    buffer.write(line)
    return tables


def update_table_header(path, suffix, headers):
    i = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            if not f.endswith(suffix):
                continue
            buffer = StringIO()
            with open(os.path.join(root, f), 'r') as fp:
                for line in fp:
                    buffer.write(line)
                    if line[0] == '<' and line[1] != '/':
                        table = line[1:line.index(':')]
                        if table in headers:
                            buffer.write('@ ' + headers[table] + '\n')
                            fp.readline()
            with open(os.path.join(root, f), 'w') as fp:
                buffer.seek(0)
                for line in buffer:
                    fp.write(line)
        i = i + 1
        if i % 100 == 0:
            print('\r%dfiles updated.' % i, end='')


if __name__ == '__main__':
    file_name = 'D:/PSASP_Pro/2020国调年度/冬低731/DataMap.txt'
    table_names = ['Grid', 'Station', 'Bus']
    with timer('Read E File repeat'):
        tables = read_efile(file_name)
    with timer('Read E File buffer'):
        tables = read_efile_buffer(file_name)


    path = 'D:/python/db/2019_09_12/2019_10_01T01_20_00/'
    headers = {'CCTOUT': 'no desc name cct gen1 gen2 times tmp1 tmp2'}
    update_table_header(path, 'res', headers)
