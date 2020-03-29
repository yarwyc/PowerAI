# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import os
import subprocess
import shutil
from multiprocessing import Pool, cpu_count

register_cmds = dict(
    psa=os.path.join(os.path.expanduser('~'), 'psa', 'bin', 'agent.sh'),
    wmlf=os.path.join(os.path.expanduser('~'), 'psa', 'bin'),
    tools=os.path.join(os.path.expanduser('~'), 'power_tools', 'bin', 'power_tools')
)


def run_cmd(args, timeout=None, cwd=None):
    return subprocess.call(args=args, timeout=timeout, cwd=cwd,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_multi_process(func, paths, n_process=None, **kwargs):
    n_process = n_process or cpu_count()
    pool = Pool(n_process)
    step = max(round(len(paths) / n_process), 1)
    for i in range(0, len(paths), step):
        pool.apply_async(func, args=(paths[i: i+step],), kwds=kwargs)
    pool.close()
    pool.join()


def call_psa(path):
    args = [register_cmds['psa'], path]
    run_cmd(args)


def call_wmlf(path):
    source = register_cmds['wmlf']
    if os.name == 'nt':
        targets = ['WMLFRTMsg.exe', 'WMLFRTMsg.dll', 'fastdb.dll',
                   'lforDLL.DLL', 'UDCore.dll']
        wmlf = 'WMLFRTMsg.exe'
    else:
        targets = ['wmlf.exe', 'wmlf.sh']
        wmlf = 'wmlf.sh'
    paths = [path] if isinstance(path, str) else path
    for p in paths:
        for t in targets:
            shutil.copy(os.path.join(source, t), os.path.join(p, t))
        run_cmd([os.path.join(p, wmlf)], cwd=p)


def call_tools(path, **kwargs):
    paths = [path] if isinstance(path, str) else path
    func = kwargs.get('sub', 0)
    for p in paths:
        args = [register_cmds['tools'], str(func), p]
        run_cmd(args)


if __name__ == '__main__':
    paths = ['D:/PSASP_Pro/wepri36_1',
             'D:/PSASP_Pro/wepri36_2',
             'D:/PSASP_Pro/wepri36_3',
             'D:/PSASP_Pro/wepri36_4']
    run_multi_process(call_wmlf, paths, 4)
    run_multi_process(call_tools, paths, 4, sub=4)