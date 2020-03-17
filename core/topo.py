# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import networkx as nx
import pandas as pd
import numpy as np
import warnings

from common.time_util import timer


class PowerGraph:
    """
    Power graph for topo analysis based on networkx.MultiGraph

    """

    def __init__(self, power, graph_type='single', node_type='bus',
                 on_only=True, edge_columns=None):
        self.node_type = node_type
        self.G = None
        if graph_type == 'single':
            self.G = self.build_graph(power, node_type, on_only)
        elif graph_type == 'multi':
            self.G = self.build_multi_graph(power, node_type, on_only, edge_columns)
        if self.G is None:
            raise NotImplementedError("Unknown type (%s, %s)" % (graph_type, node_type))
        if node_type == 'station':
            warnings.warn("Result will be wrong if there is any separate bus.")

    @staticmethod
    def build_graph(power, node_type, on_only=True):
        g = nx.Graph()
        if node_type == 'bus':
            for t in power.data['bus']['name'].items():
                g.add_node(t[0], name=t[1])
            columns = ['ibus', 'jbus', 'x', 'mark']
            values = np.vstack((power.data['acline'][columns].values,
                                power.data['transformer'][columns].values))
        elif node_type == 'station':
            for t in power.stations['name'].items():
                g.add_node(t[0], name=t[1])
            columns = ['st_i', 'st_j', 'x', 'mark']
            values = power.data['acline'][columns].values
        else:
            return None
        if len(values) == 0:
            return g
        values = values[values[:, 0] != values[:, 1]].astype(np.float)
        if on_only:
            values = values[values[:, 3] == 1.]
        values[:, :2] = np.vstack((values[:,:2].min(axis=1),
                                   values[:,:2].max(axis=1))).T
        branches = pd.DataFrame(data=values, columns=['i', 'j', 'x', 'mark'])
        branches['x'] = 1. / branches['x']
        branches[['i', 'j', 'mark']] = branches[['i', 'j', 'mark']].astype(int, copy=False)
        groups = branches.groupby(['i', 'j'], sort=False)\
            .agg({'x': 'sum', 'mark': 'count'})
        for t in groups.itertuples():
            g.add_edge(t[0][0], t[0][1], weight=1./t[1], amount=t[2])
        return g

    @staticmethod
    def build_multi_graph(power, node_type, on_only=True, edge_columns=None):
        prop_columns = edge_columns[:] if isinstance(edge_columns, list) else []
        g = nx.MultiGraph()
        if node_type == 'bus':
            for t in power.data['bus']['name'].items():
                g.add_node(t[0], name=t[1])
            columns = ['ibus', 'jbus', 'mark', 'name'] + prop_columns
            values = np.vstack((power.data['acline'][columns].values,
                                power.data['transformer'][columns].values))
        elif node_type == 'station':
            for t in power.stations['name'].items():
                g.add_node(t[0], name=t[1])
            columns = ['st_i', 'st_j', 'mark', 'name'] + prop_columns
            values = power.data['acline'][columns].values
        else:
            return None
        if len(values) == 0:
            return g
        values = values[values[:, 0] != values[:, 1]]
        if on_only:
            values = values[values[:, 2] == 1]
        for v in values:
            g.add_edge(v[0], v[1], **dict(zip(columns[3:], v[3:])))
        return g

    def get_islands(self, min_num=0):
        islands = sorted(nx.connected_components(self.G),
                         key=len, reverse=True)
        nodes = self.G.nodes()
        res = pd.Series([-1] * len(nodes), index=nodes, name='island')
        for i, nodes in enumerate(islands):
            if len(nodes) < min_num:
                break
            res[nodes] = i
        return res

    def get_bridges(self):
        res = []
        if self.G.is_multigraph():
            raise NotImplementedError("Multi Graph not supported.")
        else:
            for b in nx.bridges(self.G):
                if self.G.edges[b].get('amount', 0) == 1:
                    res.append(b)
        return res


if __name__ == '__main__':
    from core.power import Power
    path = 'D:/PSA_src/psa/localdata/0913/data'
    fmt = 'on'
    power = Power(fmt)
    power.load_power(path, fmt=fmt, lp=False, st=False)
    with timer("graph"):
        graph1 = PowerGraph(power, graph_type='single', node_type='station',
                            on_only=True)
        islands1 = graph1.get_islands(5)
        graph2 = PowerGraph(power, graph_type='multi', node_type='bus',
                            on_only=False, edge_columns=['x'])
        islands2 = graph2.get_islands(10)
