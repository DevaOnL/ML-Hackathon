"""Route-targeted transition regularisation; original DP and graph walks retained."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
import numpy as np
from .matcher import MapMatcher

@dataclass(frozen=True)
class RoutePolicy:
    reverse_penalty:float=0.0
    reverse_chord_limit_m:float=20.0
    edge_switch_penalty:float=0.0
    def __post_init__(self):
        for key,value in asdict(self).items():
            if isinstance(value,bool) or not math.isfinite(value) or value<0:raise ValueError(key+'must be finite/nonnegative')

class MemoIndex:
    def __init__(self,index):self.index=index;self.values={}
    def query(self,point,radius,count):
        key=(np.asarray(point,dtype='<f8').tobytes(),float(radius),int(count))
        if key not in self.values:self.values[key]=self.index.query(point,radius,count)
        return self.values[key]

class MemoEmissions:
    def __init__(self,model):self.model=model;self.values={};self.graph=None;self.track=None
    def score_candidates_batch(self,graph,track,indices,candidates):
        if self.graph is None:self.graph=graph;self.track=np.asarray(track).copy()
        if graph is not self.graph or not np.array_equal(track,self.track):raise ValueError('emission memo graph/track changed')
        key=(np.asarray(indices,dtype='<i8').tobytes(),tuple(id(c) for c in candidates))
        if key not in self.values:
            values=self.model.score_candidates_batch(graph,track,indices,candidates)
            if len(values)!=len(candidates):raise ValueError('probability observation count differs')
            values=[np.asarray(v,dtype=np.float64).copy() for v in values]
            for v,c in zip(values,candidates):
                if v.shape!=(len(c),) or not np.isfinite(v).all() or np.any(v<0) or not np.isclose(v.sum(),1,rtol=0,atol=1e-10):raise ValueError('invalid learned emission')
            self.values[key]=values
        return [v.copy() for v in self.values[key]]

class RouteTargetMatcher(MapMatcher):
    def __init__(self,graph,config,emission_model,policy=None,*,shared_index=None,shared_router=None,transition_cache=None):
        super().__init__(graph,config,emission_model)
        self.policy=policy or RoutePolicy()
        if shared_index is not None:self.index=shared_index
        if shared_router is not None:self.router=shared_router
        self.transition_cache=transition_cache if transition_cache is not None else {}
    def _transitions(self,previous,current,chord,horizon):
        key=(id(previous),id(current),float(chord),float(horizon),float(self.config.backward_tolerance_m))
        if key not in self.transition_cache:
            residual=super()._transitions(previous,current,chord,horizon)*self.config.beta_m
            residual.setflags(write=False);self.transition_cache[key]=residual
        cost=self.transition_cache[key]/self.config.beta_m
        a,b=previous.edges[:,None],current.edges[None,:]
        if self.policy.edge_switch_penalty:cost=cost+self.policy.edge_switch_penalty*(a!=b)
        if self.policy.reverse_penalty and chord<=self.policy.reverse_chord_limit_m:
            reverse=(a!=b)&(self.graph.tails[a]==self.graph.heads[b])&(self.graph.heads[a]==self.graph.tails[b])
            cost=cost+self.policy.reverse_penalty*reverse
        return cost
    def match(self,track):
        result=super().match(track);result.diagnostics['route_policy']=asdict(self.policy)
        result.diagnostics['postprediction_route_editing']=False
        return result
