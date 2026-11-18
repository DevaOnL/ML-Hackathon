"""In-decoder prior for exact appended arcs, retaining shortest-metre routes."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import asdict,dataclass
import math
import numpy as np
from .target96 import RouteTargetMatcher,RoutePolicy
from .routing import Router

class HopRouter(Router):
    """Read hop counts from the SAME deterministic metre-Dijkstra predecessors."""
    def __init__(self,graph,cache_size=2048):
        super().__init__(graph,cache_size);self._hops=OrderedDict()
    def hop_counts(self,source,targets,max_distance):
        distances,previous=self._search(source,max_distance)
        existing=self._hops.get(source)
        if existing is None or existing[0] is not previous:
            known={source:0};self._hops[source]=(previous,known)
        else:known=existing[1]
        self._hops.move_to_end(source)
        if len(self._hops)>self.cache_size:self._hops.popitem(last=False)
        counts=[]
        for target in targets:
            target=int(target)
            if distances.get(target,math.inf)>max_distance or not math.isfinite(distances.get(target,math.inf)):
                counts.append(math.inf);continue
            node=target;trail=[]
            while node not in known:
                trail.append(node);node=int(self.graph.tails[previous[node]])
            hops=known[node]
            for node in reversed(trail):hops+=1;known[node]=hops
            counts.append(known[target])
        return np.asarray(counts,dtype=np.float64)

@dataclass(frozen=True)
class ArcPolicy:
    appended_arc_penalty:float=0.
    reverse_penalty:float=0.
    reverse_chord_limit_m:float=20.
    edge_switch_penalty:float=0.
    def __post_init__(self):
        for name,value in asdict(self).items():
            if isinstance(value,bool) or not math.isfinite(value) or value<0:raise ValueError(name+' must be finite/nonnegative')

class ArcPriorMatcher(RouteTargetMatcher):
    def __init__(self,graph,config,emission_model,policy=None,**kwargs):
        arc_policy=policy or ArcPolicy();base_policy=RoutePolicy(reverse_penalty=arc_policy.reverse_penalty,reverse_chord_limit_m=arc_policy.reverse_chord_limit_m,edge_switch_penalty=arc_policy.edge_switch_penalty)
        super().__init__(graph,config,emission_model,base_policy,**kwargs)
        self.arc_policy=arc_policy;self._arc_counts={}
        if not isinstance(self.router,HopRouter):self.router=HopRouter(graph,config.cache_size)
    def _appended_counts(self,previous,current,horizon):
        key=(id(previous),id(current),float(horizon),float(self.config.backward_tolerance_m))
        if key not in self._arc_counts:
            graph=self.graph;e1,e2=previous.edges,current.edges
            sources,inverse=np.unique(graph.heads[e1],return_inverse=True);targets=graph.tails[e2]
            counts=np.array([self.router.hop_counts(int(s),targets,horizon) for s in sources])[inverse]+1.
            same=e1[:,None]==e2[None,:]
            along=current.fractions[None,:]*graph.edge_lengths[e2][None,:]-previous.fractions[:,None]*graph.edge_lengths[e1][:,None]
            direct=same&(along>=-self.config.backward_tolerance_m);counts[direct]=0.
            counts.setflags(write=False);self._arc_counts[key]=counts
        return self._arc_counts[key]
    def _transitions(self,previous,current,chord,horizon):
        costs=super()._transitions(previous,current,chord,horizon)
        if not self.arc_policy.appended_arc_penalty:return costs
        counts=self._appended_counts(previous,current,horizon)
        finite=np.isfinite(costs)
        if not np.isfinite(counts[finite]).all():raise RuntimeError('metre route/predecessor hop mismatch')
        costs=costs.copy();costs[finite]+=self.arc_policy.appended_arc_penalty*counts[finite]
        return costs
    def match(self,track):
        result=super().match(track);result.diagnostics['arc_policy']=asdict(self.arc_policy)
        result.diagnostics['routing_metric']='original metre Dijkstra; exact chosen connector hopprior only in Viterbi'
        return result
