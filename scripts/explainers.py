from __future__ import annotations

"""Budget-aware adaptations of established XAI families.

These online adapters explain actionable algorithm controls under a fixed
algorithmic-state snapshot. They do not treat importance as a movement
direction. Direction is estimated later by separate, paid CRN probes.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
from utils import json_safe

@dataclass
class ExplanationResult:
    method: str
    feature_names: List[str]
    importance_weights: Dict[str, float]
    interactions: Dict[str, float]
    dominant_features: List[str]
    query_count: int
    notes: str = ""
    def as_dict(self):
        d = asdict(self)
        d["importance_weights_json"] = json_safe(self.importance_weights)
        d["interactions_json"] = json_safe(self.interactions)
        return d

class EmpiricalPredictor:
    """Cached charged oracle used only during the explanation stage."""
    def __init__(self, oracle, category: str = "explanation"):
        self.oracle = oracle
        self.category = category
        self.cache: Dict[Tuple[float, ...], float] = {}
        self.query_count = 0
    def score(self, vector: Sequence[float]) -> float:
        key = tuple(round(float(x), 8) for x in vector)
        if key not in self.cache:
            self.cache[key] = float(self.oracle.score_vector(vector, category=self.category))
            self.query_count += 1
        return self.cache[key]

class BaseExplainer:
    name = "BASE"
    def required_queries_upper_bound(self, n_features: int, config) -> int:
        raise NotImplementedError
    def explain(self, predictor: EmpiricalPredictor, current: np.ndarray, bounds: np.ndarray, rng, config) -> ExplanationResult:
        raise NotImplementedError
    @staticmethod
    def _dominants(weights: Mapping[str, float], top_k: int = 3) -> List[str]:
        return [k for k, _ in sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]]

class ShapCFExplainer(BaseExplainer):
    """Budget-aware permutation-SHAP adaptation over actionable controls."""
    name = "SHAP-CF"
    def required_queries_upper_bound(self, n_features, config):
        m=max(1,int(n_features)); perms=max(1,int(config.shap_queries)//max(2,m+1)); return perms*(m+1)
    def explain(self,predictor,current,bounds,rng,config):
        names=list(self.feature_names);m=len(names);baseline=np.mean(bounds,axis=1);w=np.zeros(m)
        perms=max(1,int(config.shap_queries)//max(2,m+1))
        for _ in range(perms):
            order=rng.permutation(m);x=baseline.copy();prev=predictor.score(x)
            for j in order:
                x2=x.copy();x2[j]=current[j];new=predictor.score(x2);w[j]+=prev-new;x,prev=x2,new
        w/=max(1,perms);out=dict(zip(names,map(float,w)))
        return ExplanationResult(self.name,names,out,{},self._dominants(out),predictor.query_count,"Budget-aware bounded permutation-SHAP adaptation")

class LimeCFExplainer(BaseExplainer):
    """Budget-aware LIME adaptation with bounded empirical perturbations."""
    name="LIME-CF"
    def required_queries_upper_bound(self,n_features,config): return max(int(n_features)+2,int(config.lime_queries))
    def explain(self,predictor,current,bounds,rng,config):
        names=list(self.feature_names);m=len(names);n=self.required_queries_upper_bound(m,config);widths=np.maximum(bounds[:,1]-bounds[:,0],1e-12)
        X=[current.copy()]
        for _ in range(n-1): X.append(np.clip(rng.normal(current,.12*widths),bounds[:,0],bounds[:,1]))
        X=np.asarray(X);y=np.asarray([predictor.score(row) for row in X]);d=np.linalg.norm((X-current)/widths,axis=1);ww=np.exp(-(d**2)/.75)
        Xn=(X-current)/widths;design=np.column_stack([np.ones(len(Xn)),Xn]);ridge=1e-6*np.eye(design.shape[1]);ridge[0,0]=0
        beta=np.linalg.solve(design.T@(ww[:,None]*design)+ridge,design.T@(ww*y));out=dict(zip(names,map(float,-beta[1:])))
        return ExplanationResult(self.name,names,out,{},self._dominants(out),predictor.query_count,"Budget-aware bounded LIME adaptation")

class AcmeCFExplainer(BaseExplainer):
    """Budget-aware AcME adaptation using bounded what-if scans."""
    name="ACME-CF"
    def required_queries_upper_bound(self,n_features,config): return 1+int(n_features)*max(2,int(config.acme_grid_points_per_feature))
    def explain(self,predictor,current,bounds,rng,config):
        names=list(self.feature_names);base=predictor.score(current);points=max(2,int(config.acme_grid_points_per_feature));weights={}
        for j,name in enumerate(names):
            effects=[]
            for val in np.linspace(bounds[j,0],bounds[j,1],points):
                row=current.copy();row[j]=val;effects.append(base-predictor.score(row))
            weights[name]=float(max(effects,key=abs))
        return ExplanationResult(self.name,names,weights,{},self._dominants(weights),predictor.query_count,"Budget-aware AcME what-if adaptation")

class IBreakDownCFExplainer(BaseExplainer):
    """Budget-aware iBreakDown adaptation with bounded local pair checks."""
    name="IBREAKDOWN-CF"
    def required_queries_upper_bound(self,n_features,config):
        m=min(int(n_features),int(config.ibreakdown_max_features));return 1+m+min(int(config.ibreakdown_pair_checks),max(0,m*(m-1)//2))
    def explain(self,predictor,current,bounds,rng,config):
        names=list(self.feature_names);m=min(len(names),int(config.ibreakdown_max_features));baseline=np.mean(bounds,axis=1);full=predictor.score(current);weights={}
        for j,name in enumerate(names[:m]):
            row=current.copy();row[j]=baseline[j];weights[name]=float(predictor.score(row)-full)
        for name in names[m:]:weights[name]=0.0
        interactions={};ranked=sorted(range(m),key=lambda j:abs(weights[names[j]]),reverse=True);pairs=[]
        for a in range(len(ranked)):
            for b in range(a+1,len(ranked)):pairs.append((ranked[a],ranked[b]))
        for j,k in pairs[:int(config.ibreakdown_pair_checks)]:
            row=current.copy();row[j]=baseline[j];row[k]=baseline[k];combined=float(predictor.score(row)-full);interactions[f"{names[j]}*{names[k]}"]=combined-weights[names[j]]-weights[names[k]]
        return ExplanationResult(self.name,names,weights,interactions,self._dominants(weights),predictor.query_count,"Budget-aware iBreakDown adaptation")

EXPLAINERS={"SHAP-CF":ShapCFExplainer,"LIME-CF":LimeCFExplainer,"ACME-CF":AcmeCFExplainer,"IBREAKDOWN-CF":IBreakDownCFExplainer}
def make_explainer(variant:str,feature_names:Sequence[str])->BaseExplainer:
    if variant not in EXPLAINERS:raise KeyError(f"Unsupported XAI variant: {variant}")
    obj=EXPLAINERS[variant]();obj.feature_names=list(feature_names);return obj
