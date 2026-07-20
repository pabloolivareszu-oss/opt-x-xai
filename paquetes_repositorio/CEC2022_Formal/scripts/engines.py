from __future__ import annotations

"""Strict-budget v7.3 engines.

Every real objective call is ledgered. XAI importance and parameter movement
direction are separate. Probe comparisons use common random numbers (CRN).
Trajectories are streamed incrementally for selected runs.
"""
from dataclasses import dataclass, asdict
import math, time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import scipy.stats as stats
from budget import StrictEvaluationBudget, TargetReached
from config import OracleConfig
from cec2022_protocol import TARGET_ERROR
from explainers import EmpiricalPredictor, make_explainer
from trajectory import IncrementalTrajectoryWriter
from utils import json_safe, population_diversity, relative_improvement, safe_error, slope, stable_seed

@dataclass
class InterventionLog:
    Dimension:int; Function:int; Algorithm:str; Variant:str; Run:int
    Trigger_TotalFEs:int; RemainingFEs_Before:int; RequiredFEs_Phase:int
    Activated:bool; Skipped_By_Budget:bool
    ExplanationFEs:int; DirectionProbeFEs:int; CFValidationFEs:int; TotalOracleFEs:int
    RemainingFEs_After:int
    History_Window_FEs:int; History_Snapshots:int
    Actionable_Params_Before_JSON:str; Observable_State_JSON:str; Full_Algorithmic_State_JSON:str
    Importance_Weights_JSON:str; Normalized_Actionable_Weights_JSON:str
    Interaction_Weights_JSON:str; Dominant_Features_JSON:str
    Direction_Scores_JSON:str; Preferred_Directions_JSON:str; Direction_Source_JSON:str
    CF_Generated:int; CF_Validated:int; CF_Configuration_JSON:str; Actionable_Params_After_JSON:str
    Intervention_Distance:float
    Fitness_Before_Phase:float; Fitness_After_Explanation:float; Gain_Explanation:float
    Fitness_After_Direction_Probe:float; Gain_Direction_Probe:float
    Fitness_After_CF_Validation:float; Gain_CF_Validation:float
    Baseline_Probe_Fit:float; Best_CF_Probe_Fit:float; Validated_Improvement:float
    Applied:bool; Diversity_Pre:float; DiversityNorm_Pre:float
    XAI_Interval_Start_FE:int; Explanation_Interval_End_FE:int; Direction_Interval_End_FE:int; XAI_Interval_End_FE:int
    Explanation_Time_ms:float; Direction_Time_ms:float; CF_Time_ms:float; Status:str
    Stagnation_Episode_Anchor_FE:int=0
    Fitness_Post_Window:float=math.nan; Error_Post_Window:float=math.nan
    Diversity_Post:float=math.nan; DiversityNorm_Post:float=math.nan
    Realized_Intervention_Gain:float=math.nan; Delta_DiversityNorm:float=math.nan
    Useful:bool=False; Noise_Failure:bool=False
    def as_dict(self): return asdict(self)

@dataclass
class ProbeAnchor:
    fe:int
    population:np.ndarray
    fitness:np.ndarray

class ProbeOracle:
    """Paid retrospective short-horizon oracle with CRN across vectors.

    The score is the median response over a bounded set of recent trajectory
    snapshots. Every real objective call is charged as ExplanationFEs.
    """
    def __init__(self,engine:'BaseEngine',config:OracleConfig,seed:int):
        self.engine=engine; self.config=config; self.seed=int(seed); self.anchors=[]
        for anchor in engine.history_anchors():
            order=np.argsort(anchor.fitness); cap=min(len(order),int(config.probe_population_cap))
            idx=order if cap>=len(order) else order[np.linspace(0,len(order)-1,cap).astype(int)]
            self.anchors.append(ProbeAnchor(int(anchor.fe),anchor.population[idx].copy(),anchor.fitness[idx].copy()))
        if not self.anchors:
            self.anchors=[ProbeAnchor(int(engine.budget.total_fes),engine.population.copy(),engine.fitness.copy())]
    @property
    def probe_cost(self):
        return int(sum(len(a.population) for a in self.anchors)*self.config.probe_iters*self.config.probe_repeats)
    def score_vector(self,vector:Sequence[float],*,category:str='explanation')->float:
        params=self.engine.vector_to_params(vector); vals=[]
        for ai,anchor in enumerate(self.anchors):
            for rep in range(int(self.config.probe_repeats)):
                rng=np.random.default_rng(stable_seed(self.seed,"CRN",ai,rep))
                pop=anchor.population.copy(); fit=anchor.fitness.copy(); best=float(np.min(fit))
                for _ in range(int(self.config.probe_iters)):
                    candidates=self.engine.probe_candidates(pop,fit,params,rng)
                    for i,cand in enumerate(candidates):
                        value=self.engine.budget.evaluate(cand,category=category)
                        if value<fit[i]: pop[i]=cand; fit[i]=value
                        if value<best: best=float(value)
                vals.append(best)
        return float(np.median(vals))

class BaseEngine:
    algorithm_name="BASE"; canonical_status="reference"
    def __init__(self,func_obj,dim:int,max_fes:int,variant:str,oracle_config:OracleConfig,*,run_id:int,function_id:int,seed:int,trajectory_dir=None,seed_mode:str="paired"):
        self.func_obj=func_obj; self.dim=int(dim); self.lb=np.broadcast_to(np.asarray(func_obj.lb,float),(self.dim,)).copy(); self.ub=np.broadcast_to(np.asarray(func_obj.ub,float),(self.dim,)).copy(); self.optimum=float(func_obj.f_global)
        self.max_fes=int(max_fes); self.variant=variant; self.oracle_config=oracle_config; self.run_id=int(run_id); self.function_id=int(function_id); self.base_seed=int(seed); self.seed_mode=str(seed_mode); self.seed=int(seed)
        self.rng=np.random.default_rng(self.seed if self.seed_mode=="official" else stable_seed(seed,"optimizer")); self.oracle_rng=np.random.default_rng((self.seed+1) if self.seed_mode=="official" else stable_seed(seed,variant,"oracle"))
        self.generation=0; self.oracle_cooldown_until_fe=0; self.oracle_attempts=0; self.oracle_activations=0; self.interventions_applied=0; self.rescues=0; self.failed_interventions=0; self.budget_skips=0; self.oracle_time_ms=0.0
        self.pending_interventions:List[Tuple[int,int]]=[]; self.intervention_logs:List[InterventionLog]=[]; self.param_trace=[]; self.recent_success_rates=[]; self.recent_diversity_norm=[]; self.recent_errors=[]; self.override_params=None; self.override_until_fe=0
        self.recent_snapshots:List[ProbeAnchor]=[]; self.successful_parameter_history=[]
        self.budget=StrictEvaluationBudget(func_obj.evaluate,self.max_fes,self.optimum,meaningful_optimizer_improvement_rel_threshold=self.oracle_config.meaningful_optimizer_improvement_rel_threshold,target_error=TARGET_ERROR); self.trajectory_writer=IncrementalTrajectoryWriter(trajectory_dir)
        self._initialize_algorithm(); self.initial_diversity,self.initial_diversity_norm=population_diversity(self.population,self.lb,self.ub); self.recent_diversity_norm.append(self.initial_diversity_norm); self.recent_errors.append(self.current_error); self._remember_snapshot(); self.trajectory_writer.write(self.population,0,self.budget)
    def _initialize_algorithm(self): raise NotImplementedError
    def actionable_parameters(self)->Dict[str,float]: raise NotImplementedError
    def actionable_bounds(self)->Dict[str,Tuple[float,float]]: raise NotImplementedError
    def optimizer_generation(self)->Tuple[int,int]: raise NotImplementedError
    def probe_candidates(self,pop,fit,params,rng)->np.ndarray: raise NotImplementedError
    @property
    def trigger_window_fes(self): return max(1,int(round(self.max_fes*self.oracle_config.trigger_window_fraction)))
    @property
    def history_window_fes(self): return max(1,int(round(self.max_fes*self.oracle_config.history_window_fraction)))
    @property
    def post_window_fes(self): return max(1,int(round(self.max_fes*self.oracle_config.post_window_fraction)))
    @property
    def cooldown_fes(self): return max(1,int(round(self.max_fes*self.oracle_config.cooldown_fraction)))
    @property
    def stagnation_fes(self): return int(self.budget.total_fes-self.budget.last_meaningful_optimizer_improvement_fe)
    @property
    def stagnation_episode_anchor_fe(self): return int(self.budget.last_meaningful_optimizer_improvement_fe)
    def apply_intervention(self,params):
        # Persistent parameter update until the next XAI activation. No restart.
        self.override_params={k:float(v) for k,v in params.items()}; self.override_until_fe=self.max_fes
    def current_parameters(self):
        base=self.actionable_parameters()
        if self.override_params is not None:
            out=dict(base); out.update(self.override_params); return out
        return base
    @property
    def best_fit(self): return float(self.budget.best_fitness)
    @property
    def current_error(self): return safe_error(self.best_fit,self.optimum)
    def params_to_vector(self,params): return np.asarray([float(params[k]) for k in self.actionable_bounds()],float)
    def vector_to_params(self,vector): return {k:float(v) for k,v in zip(self.actionable_bounds(),vector)}
    def bounds_array(self): return np.asarray([self.actionable_bounds()[k] for k in self.actionable_bounds()],float)
    def observable_state(self):
        raw,divn=population_diversity(self.population,self.lb,self.ub)
        return {"budget_progress":self.budget.total_fes/max(1,self.max_fes),"remaining_fes":float(self.budget.remaining_fes),"stagnation_fes":float(self.stagnation_fes),"last_global_improvement_fe":float(self.budget.last_improvement_fe),"last_optimizer_improvement_fe":float(self.budget.last_optimizer_improvement_fe),"last_meaningful_optimizer_improvement_fe":float(self.budget.last_meaningful_optimizer_improvement_fe),"best_optimizer_fitness":float(self.budget.best_optimizer_fitness),"last_meaningful_optimizer_fitness":float(self.budget.last_meaningful_optimizer_fitness),"recent_success_rate":float(np.mean(self.recent_success_rates[-5:])) if self.recent_success_rates else 1.0,"diversity":raw,"diversity_norm":divn,"diversity_ratio":divn/max(self.initial_diversity_norm,1e-12),"diversity_slope":slope(self.recent_diversity_norm,5),"relative_error_improvement":relative_improvement(self.recent_errors,10),"population_size":float(len(self.population)),"population_size_ratio":len(self.population)/max(1,getattr(self,"initial_population_size",len(self.population))),"oracle_cooldown_remaining_fes":float(max(0,self.oracle_cooldown_until_fe-self.budget.total_fes)),"history_window_fes":float(self.history_window_fes),"history_snapshot_count":float(len(self.history_anchors()))}
    def algorithm_state(self): return {**{f"param_{k}":float(v) for k,v in self.current_parameters().items()},**{f"observable_{k}":float(v) for k,v in self.observable_state().items()}}
    def surgical_trigger(self):
        if self.budget.total_fes < self.oracle_cooldown_until_fe: return False
        return bool(self.stagnation_fes >= self.trigger_window_fes)
    def _remember_snapshot(self):
        self.recent_snapshots.append(ProbeAnchor(int(self.budget.total_fes),self.population.copy(),self.fitness.copy()))
        floor=self.budget.total_fes-self.history_window_fes
        self.recent_snapshots=[s for s in self.recent_snapshots if s.fe>=floor][-int(self.oracle_config.max_history_buffer_snapshots):]
    def history_anchors(self):
        snaps=self.recent_snapshots or [ProbeAnchor(int(self.budget.total_fes),self.population.copy(),self.fitness.copy())]
        k=min(len(snaps),max(1,int(self.oracle_config.history_snapshots)))
        idx=np.linspace(0,len(snaps)-1,k).astype(int)
        return [snaps[i] for i in sorted(set(idx.tolist()))]
    def _explainer_query_upper(self): return make_explainer(self.variant,list(self.actionable_bounds())).required_queries_upper_bound(len(self.actionable_bounds()),self.oracle_config)
    def _phase_required_fes(self,probe_cost): return int(self._explainer_query_upper()*probe_cost)
    def _historical_directions(self,params):
        floor=self.budget.total_fes-self.history_window_fes; dirs={}; sources={}
        hist=[row for row in self.successful_parameter_history if row[0]>=floor]
        for j,name in enumerate(self.actionable_bounds()):
            vals=[float(row[1].get(name,params[name])) for row in hist if name in row[1]]
            delta=(float(np.median(vals))-float(params[name])) if vals else 0.0
            if abs(delta)>1e-15:
                dirs[name]=1 if delta>0 else -1; sources[name]="recent_success_history"
            else:
                dirs[name]=1 if stable_seed(self.seed,self.variant,"fallback_direction",self.oracle_activations,name)%2==0 else -1; sources[name]="deterministic_fallback"
        return dirs,sources
    def _deterministic_counterfactual(self,params,importance,directions):
        bounds=self.actionable_bounds(); names=list(bounds); mags=np.asarray([abs(float(importance.get(k,0.0))) for k in names],float)
        if float(mags.sum())<=1e-15: mags=np.ones(len(names),float)
        weights=mags/mags.sum(); out={}; distance=0.0
        for name,w in zip(names,weights):
            lo,hi=bounds[name]; width=max(hi-lo,1e-12); step=float(directions[name])*float(self.oracle_config.intervention_strength)*float(w)*width
            out[name]=float(np.clip(float(params[name])+step,lo,hi)); distance+=abs(out[name]-float(params[name]))/width
        return out,dict(zip(names,map(float,weights))),float(distance)
    def _empty_log(self,base,params,*,status,sb=False):
        return InterventionLog(**base,Activated=False,Skipped_By_Budget=sb,ExplanationFEs=0,DirectionProbeFEs=0,CFValidationFEs=0,TotalOracleFEs=0,RemainingFEs_After=self.budget.remaining_fes,History_Window_FEs=self.history_window_fes,History_Snapshots=0,Importance_Weights_JSON="{}",Normalized_Actionable_Weights_JSON="{}",Interaction_Weights_JSON="{}",Dominant_Features_JSON="[]",Direction_Scores_JSON="{}",Preferred_Directions_JSON="{}",Direction_Source_JSON="{}",CF_Generated=0,CF_Validated=0,CF_Configuration_JSON=json_safe(params),Actionable_Params_After_JSON=json_safe(params),Intervention_Distance=0.0,Fitness_After_Explanation=self.best_fit,Gain_Explanation=0.0,Fitness_After_Direction_Probe=self.best_fit,Gain_Direction_Probe=0.0,Fitness_After_CF_Validation=self.best_fit,Gain_CF_Validation=0.0,Baseline_Probe_Fit=math.nan,Best_CF_Probe_Fit=math.nan,Validated_Improvement=math.nan,Applied=False,Explanation_Interval_End_FE=self.budget.total_fes,Direction_Interval_End_FE=self.budget.total_fes,XAI_Interval_End_FE=self.budget.total_fes,Explanation_Time_ms=0.0,Direction_Time_ms=0.0,CF_Time_ms=0.0,Status=status)
    def maybe_run_oracle(self):
        # After reaching the CEC2022 target, preserve exact fixed-budget accounting
        # but do not spend additional FEs on explanations.
        if self.budget.first_target_fe > 0: return
        if self.variant=="Standard" or not self.surgical_trigger(): return
        anchor=self.stagnation_episode_anchor_fe; params=self.current_parameters(); state=self.algorithm_state(); obs=self.observable_state(); start_fe=self.budget.total_fes; raw,divn=population_diversity(self.population,self.lb,self.ub); fit_before=self.best_fit; remaining=self.budget.remaining_fes
        self.oracle_attempts+=1; probe_seed=stable_seed(self.seed,self.variant,self.budget.total_fes,"probe"); oracle=ProbeOracle(self,self.oracle_config,probe_seed); required=self._phase_required_fes(oracle.probe_cost)
        base=dict(Dimension=self.dim,Function=self.function_id,Algorithm=self.algorithm_name,Variant=self.variant,Run=self.run_id,Trigger_TotalFEs=start_fe,RemainingFEs_Before=remaining,RequiredFEs_Phase=required,Actionable_Params_Before_JSON=json_safe(params),Observable_State_JSON=json_safe(obs),Full_Algorithmic_State_JSON=json_safe(state),Fitness_Before_Phase=fit_before,Diversity_Pre=raw,DiversityNorm_Pre=divn,XAI_Interval_Start_FE=start_fe,Stagnation_Episode_Anchor_FE=anchor)
        if remaining<required:
            self.budget_skips+=1
            self.oracle_cooldown_until_fe=min(self.max_fes,self.budget.total_fes+max(1,self.cooldown_fes//3))
            self.intervention_logs.append(self._empty_log(base,params,status="SKIPPED_BY_GLOBAL_FE_AVAILABILITY",sb=True))
            return
        self.oracle_activations+=1; exp0=self.budget.explanation_fes
        t=time.perf_counter(); predictor=EmpiricalPredictor(oracle,"explanation"); explainer=make_explainer(self.variant,list(self.actionable_bounds()))
        try:
            res=explainer.explain(predictor,self.params_to_vector(params),self.bounds_array(),self.oracle_rng,self.oracle_config)
        except TargetReached:
            exp_ms=(time.perf_counter()-t)*1000
            partial=self._empty_log(base,params,status="TARGET_REACHED_DURING_EXPLANATION")
            partial.Activated=True
            partial.ExplanationFEs=int(self.budget.explanation_fes-exp0)
            partial.TotalOracleFEs=int(self.budget.explanation_fes-exp0)
            partial.RemainingFEs_After=int(self.budget.remaining_fes)
            partial.Fitness_After_Explanation=float(self.best_fit)
            partial.Gain_Explanation=float(fit_before-self.best_fit)
            partial.Explanation_Interval_End_FE=int(self.budget.total_fes)
            partial.Direction_Interval_End_FE=int(self.budget.total_fes)
            partial.XAI_Interval_End_FE=int(self.budget.total_fes)
            partial.Explanation_Time_ms=round(exp_ms,4)
            self.intervention_logs.append(partial)
            self.oracle_time_ms+=exp_ms
            return
        exp_ms=(time.perf_counter()-t)*1000; fit_after_exp=self.best_fit; exp_end=self.budget.total_fes
        if self.budget.first_target_fe>0: return
        directions,sources=self._historical_directions(params); cf,norm_weights,distance=self._deterministic_counterfactual(params,res.importance_weights,directions); self.apply_intervention(cf); self.interventions_applied+=1; self.oracle_cooldown_until_fe=min(self.max_fes,self.budget.total_fes+self.cooldown_fes)
        log=InterventionLog(**base,Activated=True,Skipped_By_Budget=False,ExplanationFEs=self.budget.explanation_fes-exp0,DirectionProbeFEs=0,CFValidationFEs=0,TotalOracleFEs=self.budget.explanation_fes-exp0,RemainingFEs_After=self.budget.remaining_fes,History_Window_FEs=self.history_window_fes,History_Snapshots=len(oracle.anchors),Importance_Weights_JSON=json_safe(res.importance_weights),Normalized_Actionable_Weights_JSON=json_safe(norm_weights),Interaction_Weights_JSON=json_safe(res.interactions),Dominant_Features_JSON=json_safe(res.dominant_features),Direction_Scores_JSON="{}",Preferred_Directions_JSON=json_safe(directions),Direction_Source_JSON=json_safe(sources),CF_Generated=1,CF_Validated=0,CF_Configuration_JSON=json_safe(cf),Actionable_Params_After_JSON=json_safe(cf),Intervention_Distance=distance,Fitness_After_Explanation=fit_after_exp,Gain_Explanation=fit_before-fit_after_exp,Fitness_After_Direction_Probe=fit_after_exp,Gain_Direction_Probe=0.0,Fitness_After_CF_Validation=fit_after_exp,Gain_CF_Validation=0.0,Baseline_Probe_Fit=math.nan,Best_CF_Probe_Fit=math.nan,Validated_Improvement=math.nan,Applied=True,Explanation_Interval_End_FE=exp_end,Direction_Interval_End_FE=exp_end,XAI_Interval_End_FE=self.budget.total_fes,Explanation_Time_ms=round(exp_ms,4),Direction_Time_ms=0.0,CF_Time_ms=0.0,Status="APPLIED_IMMEDIATE")
        self.intervention_logs.append(log); self.pending_interventions.append((len(self.intervention_logs)-1,min(self.max_fes,self.budget.total_fes+self.post_window_fes))); self.oracle_time_ms+=exp_ms
    def _audit_pending(self):
        keep=[]
        for idx,due in self.pending_interventions:
            if self.budget.total_fes<due: keep.append((idx,due)); continue
            self._finalize_log(idx)
        self.pending_interventions=keep
    def _finalize_log(self,idx):
        log=self.intervention_logs[idx]; raw,divn=population_diversity(self.population,self.lb,self.ub); log.Fitness_Post_Window=self.best_fit; log.Error_Post_Window=self.current_error; log.Diversity_Post=raw; log.DiversityNorm_Post=divn; log.Realized_Intervention_Gain=float(log.Fitness_After_Explanation-self.best_fit); log.Delta_DiversityNorm=float(divn-log.DiversityNorm_Pre); log.Useful=bool(log.Realized_Intervention_Gain>1e-12); log.Noise_Failure=not log.Useful; self.rescues+=int(log.Useful); self.failed_interventions+=int(not log.Useful)
    def _record_trace(self):
        raw,divn=population_diversity(self.population,self.lb,self.ub); s=self.budget.snapshot(); self.param_trace.append({"Generation":self.generation,"TotalFEs":s.total_fes,"OptimizerFEs":s.optimizer_fes,"ExplanationFEs":s.explanation_fes,"DirectionProbeFEs":s.direction_probe_fes,"CFValidationFEs":s.cf_validation_fes,"Best_Fit":self.best_fit,"Best_Err":self.current_error,"Diversity":raw,"Diversity_Norm":divn,"StagnationFEs":self.stagnation_fes,"LastGlobalImprovementFE":self.budget.last_improvement_fe,"LastOptimizerImprovementFE":self.budget.last_optimizer_improvement_fe,"LastMeaningfulOptimizerImprovementFE":self.budget.last_meaningful_optimizer_improvement_fe,"BestOptimizerFit":self.budget.best_optimizer_fitness,"LastMeaningfulOptimizerFit":self.budget.last_meaningful_optimizer_fitness,"OracleCooldownRemainingFEs":max(0,self.oracle_cooldown_until_fe-self.budget.total_fes) ,"StagnationEpisodeAnchorFE":self.stagnation_episode_anchor_fe,**{f"Param_{k}":float(v) for k,v in self.current_parameters().items()}})
    def optimize(self):
        while self.budget.total_fes<self.max_fes and self.budget.first_target_fe==0:
            self.generation+=1; self.maybe_run_oracle()
            if self.budget.first_target_fe>0 or self.budget.exhausted:
                self._remember_snapshot(); self._audit_pending(); self._record_trace(); self.trajectory_writer.write(self.population,self.generation,self.budget)
                break
            success,attempted=self.optimizer_generation(); self.recent_success_rates.append(success/max(1,attempted)); _,divn=population_diversity(self.population,self.lb,self.ub); self.recent_diversity_norm.append(divn); self.recent_errors.append(self.current_error)
            if success>0: self.successful_parameter_history.append((int(self.budget.total_fes),dict(self.current_parameters())))
            self._remember_snapshot(); self._audit_pending(); self._record_trace(); self.trajectory_writer.write(self.population,self.generation,self.budget)
        for idx,_ in self.pending_interventions: self._finalize_log(idx)
        self.pending_interventions=[]; self.budget.assert_valid(require_exhausted=False); raw,divn=population_diversity(self.population,self.lb,self.ub)
        return {"final_fit":self.best_fit,"final_err":self.current_error,"best_x":(self.budget.best_x.tolist() if self.budget.best_x is not None else None),"budget":self.budget.as_dict(),"history":list(self.budget.history),"intervention_logs":[x.as_dict() for x in self.intervention_logs],"param_trace":self.param_trace,"trajectory_count":self.trajectory_writer.count,"oracle_attempts":self.oracle_attempts,"oracle_activations":self.oracle_activations,"interventions_applied":self.interventions_applied,"rescues":self.rescues,"failed_interventions":self.failed_interventions,"budget_skips":self.budget_skips,"oracle_time_ms":self.oracle_time_ms,"final_diversity":raw,"final_diversity_norm":divn}
    def _evaluate_initial_population(self,population):
        values=[]
        for x in population:
            if self.budget.first_target_fe>0 or self.budget.exhausted: break
            values.append(self.budget.evaluate(x,category="optimizer"))
        if len(values)<len(population):
            fill=float(self.budget.best_fitness)
            values.extend([fill]*(len(population)-len(values)))
        return np.asarray(values,float)
    def _evaluate_optimizer_candidates(self,candidates):
        limit=min(len(candidates),self.budget.remaining_fes); values=[]
        for i in range(limit):
            if self.budget.first_target_fe>0 or self.budget.exhausted: break
            values.append(self.budget.evaluate(candidates[i],category="optimizer"))
        return np.asarray(values,float),len(values)
class PSOEngine(BaseEngine):
    algorithm_name = "PSO"
    canonical_status = "reference_style"
    def _initialize_algorithm(self):
        self.initial_population_size = max(20, 10 * self.dim)
        self.population = self.rng.uniform(self.lb, self.ub, (self.initial_population_size, self.dim))
        span = self.ub - self.lb
        self.velocity = self.rng.uniform(-0.1 * span, 0.1 * span, self.population.shape)
        self.fitness = self._evaluate_initial_population(self.population)
        self.pbest = self.population.copy(); self.pbest_fit = self.fitness.copy()
    def actionable_parameters(self): return {"w": 0.72, "c1": 1.49, "c2": 1.49, "vmax": 0.20}
    def actionable_bounds(self): return {"w": (0.2, 0.95), "c1": (0.2, 2.5), "c2": (0.2, 2.5), "vmax": (0.05, 0.5)}
    def probe_candidates(self, pop, fit, params, rng):
        g = pop[int(np.argmin(fit))]; span = self.ub - self.lb
        vel = rng.uniform(-params["vmax"] * span, params["vmax"] * span, pop.shape)
        r1 = rng.random(pop.shape); r2 = rng.random(pop.shape)
        cand = pop + params["w"] * vel + params["c1"] * r1 * (g - pop) + params["c2"] * r2 * (g - pop)
        return np.clip(cand, self.lb, self.ub)
    def optimizer_generation(self):
        p = self.current_parameters(); g = self.pbest[int(np.argmin(self.pbest_fit))]; span = self.ub - self.lb
        r1 = self.rng.random(self.population.shape); r2 = self.rng.random(self.population.shape)
        new_v = p["w"] * self.velocity + p["c1"] * r1 * (self.pbest - self.population) + p["c2"] * r2 * (g - self.population)
        new_v = np.clip(new_v, -p["vmax"] * span, p["vmax"] * span)
        candidates = np.clip(self.population + new_v, self.lb, self.ub)
        vals, n = self._evaluate_optimizer_candidates(candidates)
        successes = 0
        for i in range(n):
            self.velocity[i] = new_v[i]; self.population[i] = candidates[i]; self.fitness[i] = vals[i]
            if vals[i] < self.pbest_fit[i]: self.pbest[i] = candidates[i]; self.pbest_fit[i] = vals[i]; successes += 1
        return successes, n


class BAEngine(BaseEngine):
    algorithm_name = "BA"
    canonical_status = "reference_style"
    def _initialize_algorithm(self):
        self.initial_population_size = max(20, 8 * self.dim)
        self.population = self.rng.uniform(self.lb, self.ub, (self.initial_population_size, self.dim))
        self.velocity = np.zeros_like(self.population)
        self.fitness = self._evaluate_initial_population(self.population)
    def actionable_parameters(self): return {"fmin": 0.0, "fmax": 2.0, "loudness": 0.9, "pulse_rate": 0.5, "alpha": 0.95, "gamma": 0.9}
    def actionable_bounds(self): return {"fmin": (0.0, 1.0), "fmax": (1.0, 3.0), "loudness": (0.1, 1.0), "pulse_rate": (0.05, 0.95), "alpha": (0.80, 0.999), "gamma": (0.1, 1.5)}
    def probe_candidates(self, pop, fit, p, rng):
        best = pop[int(np.argmin(fit))]; freq = rng.uniform(p["fmin"], p["fmax"], len(pop))[:, None]
        vel = (pop - best) * freq
        cand = pop + vel
        effective_pulse = np.clip(p["pulse_rate"] * (1.0 - math.exp(-p["gamma"] * max(1, self.generation))), 0.0, 1.0)
        effective_loudness = np.clip(p["loudness"] * p["alpha"], 0.0, 1.0)
        local = rng.random(len(pop)) > effective_pulse
        cand[local] = best + rng.normal(0, 0.02 * effective_loudness * np.mean(self.ub - self.lb), (local.sum(), self.dim))
        return np.clip(cand, self.lb, self.ub)
    def optimizer_generation(self):
        p = self.current_parameters(); candidates = self.probe_candidates(self.population, self.fitness, p, self.rng)
        vals, n = self._evaluate_optimizer_candidates(candidates); successes = 0
        for i in range(n):
            if vals[i] <= self.fitness[i] and self.rng.random() < np.clip(p["loudness"] * p["alpha"], 0.0, 1.0):
                self.population[i] = candidates[i]; self.fitness[i] = vals[i]; successes += 1
        return successes, n


class GWOEngine(BaseEngine):
    algorithm_name = "GWO"
    canonical_status = "reference_style"
    def _initialize_algorithm(self):
        self.initial_population_size = max(20, 8 * self.dim)
        self.population = self.rng.uniform(self.lb, self.ub, (self.initial_population_size, self.dim))
        self.fitness = self._evaluate_initial_population(self.population)
    def actionable_parameters(self): return {"a_scale": 1.0, "leader_alpha": 1.0, "leader_beta": 1.0, "leader_delta": 1.0}
    def actionable_bounds(self): return {"a_scale": (0.4, 1.4), "leader_alpha": (0.4, 1.6), "leader_beta": (0.4, 1.6), "leader_delta": (0.4, 1.6)}
    def probe_candidates(self, pop, fit, p, rng):
        order = np.argsort(fit); leaders = pop[order[:3]]
        a = 2.0 * p["a_scale"] * max(0.0, 1.0 - self.budget.total_fes / max(1, self.max_fes))
        outs = []
        weights = [p["leader_alpha"], p["leader_beta"], p["leader_delta"]]
        for x in pop:
            xs=[]
            for leader,w in zip(leaders,weights):
                A=2*a*rng.random(self.dim)-a; C=2*rng.random(self.dim)
                xs.append(w*(leader-A*np.abs(C*leader-x)))
            outs.append(np.sum(xs,axis=0)/max(sum(weights),1e-12))
        return np.clip(np.asarray(outs), self.lb, self.ub)
    def optimizer_generation(self):
        c = self.probe_candidates(self.population,self.fitness,self.current_parameters(),self.rng)
        vals,n=self._evaluate_optimizer_candidates(c); successes=0
        for i in range(n):
            if vals[i] < self.fitness[i]: self.population[i]=c[i]; self.fitness[i]=vals[i]; successes+=1
        return successes,n


class GSKEngine(BaseEngine):
    algorithm_name = "GSK"
    canonical_status = "reference_style"
    def _initialize_algorithm(self):
        self.initial_population_size = 100
        self.population=self.rng.uniform(self.lb,self.ub,(self.initial_population_size,self.dim)); self.fitness=self._evaluate_initial_population(self.population)
    def actionable_parameters(self): return {"KF":0.5,"KR":0.9}
    def actionable_bounds(self): return {"KF":(0.1,1.0),"KR":(0.1,0.9)}
    def probe_candidates(self,pop,fit,p,rng):
        order=np.argsort(fit); sp=pop[order]; n=len(pop); dj=np.array([int(self.dim*((n-i)/n)**p["KR"]) for i in range(n)])
        out=sp.copy()
        for i in range(n):
            x=sp[i].copy(); d=dj[i]
            if d>0:
                ib=max(0,i-1); iw=min(n-1,i+1); ir=int(rng.integers(0,n)); x[:d]=sp[i,:d]+p["KF"]*((sp[ib,:d]-sp[iw,:d])+(sp[ir,:d]-sp[i,:d]))
            if d<self.dim:
                a=max(1,int(n*.1)); b=max(a+1,int(n*.9)); ib=int(rng.integers(0,a)); im=int(rng.integers(a,b)); iw=int(rng.integers(b,n)); x[d:]=sp[i,d:]+p["KF"]*((sp[ib,d:]-sp[iw,d:])+(sp[im,d:]-sp[i,d:]))
            out[i]=np.clip(x,self.lb,self.ub)
        return out
    def optimizer_generation(self):
        c=self.probe_candidates(self.population,self.fitness,self.current_parameters(),self.rng); vals,n=self._evaluate_optimizer_candidates(c); successes=0
        order=np.argsort(self.fitness); sp=self.population[order].copy(); sf=self.fitness[order].copy()
        for i in range(n):
            if vals[i]<sf[i]: sp[i]=c[i]; sf[i]=vals[i]; successes+=1
        self.population,self.fitness=sp,sf; return successes,n


class LSHADEEngine(BaseEngine):
    algorithm_name="L-SHADE"
    canonical_status="reference_style"
    def _initialize_algorithm(self):
        self.initial_population_size=18*self.dim; self.min_population_size=4
        self.population=self.rng.uniform(self.lb,self.ub,(self.initial_population_size,self.dim)); self.fitness=self._evaluate_initial_population(self.population)
        self.m_f=np.full(6,.5); self.m_cr=np.full(6,.5); self.memory_index=0; self.archive=[]
    def actionable_parameters(self): return {"F":float(np.mean(self.m_f)),"CR":float(np.mean(self.m_cr)),"pbest_rate":0.11,"archive_rate":1.0}
    def actionable_bounds(self): return {"F":(0.1,1.0),"CR":(0.0,1.0),"pbest_rate":(0.05,0.30),"archive_rate":(0.0,2.0)}
    def probe_candidates(self,pop,fit,p,rng):
        n=len(pop); order=np.argsort(fit); out=[]
        for i in range(n):
            pl=max(2,int(n*p["pbest_rate"])); pb=pop[order[int(rng.integers(0,pl))]]; r1,r2=rng.choice(n,2,replace=False)
            mutant=pop[i]+p["F"]*(pb-pop[i])+p["F"]*p["archive_rate"]*(pop[r1]-pop[r2]); mask=rng.random(self.dim)<=p["CR"]; mask[int(rng.integers(0,self.dim))]=True
            out.append(np.clip(np.where(mask,mutant,pop[i]),self.lb,self.ub))
        return np.asarray(out)
    def _reduce_population(self):
        target=max(self.min_population_size,int(round(((self.min_population_size-self.initial_population_size)/self.max_fes)*self.budget.total_fes+self.initial_population_size)))
        if len(self.population)>target:
            bad=np.argsort(self.fitness)[-(len(self.population)-target):]; self.population=np.delete(self.population,bad,axis=0); self.fitness=np.delete(self.fitness,bad)
    def optimizer_generation(self):
        p=self.current_parameters(); c=self.probe_candidates(self.population,self.fitness,p,self.rng); vals,n=self._evaluate_optimizer_candidates(c); successes=0; diffs=[]
        for i in range(n):
            if vals[i]<=self.fitness[i]:
                d=self.fitness[i]-vals[i]; self.population[i]=c[i]; self.fitness[i]=vals[i]
                if d>0: successes+=1; diffs.append(d)
        if diffs:
            self.m_f[self.memory_index]=.5*self.m_f[self.memory_index]+.5*p["F"]; self.m_cr[self.memory_index]=.5*self.m_cr[self.memory_index]+.5*p["CR"]; self.memory_index=(self.memory_index+1)%len(self.m_f)
        self._reduce_population(); return successes,n


class jSOEngine(LSHADEEngine):
    algorithm_name="jSO"
    canonical_status="reference_style"
    def _initialize_algorithm(self):
        raw=int(25*math.log(self.dim)*math.sqrt(self.dim)); self.initial_population_size=max(18*self.dim,raw,4); self.min_population_size=4
        self.population=self.rng.uniform(self.lb,self.ub,(self.initial_population_size,self.dim)); self.fitness=self._evaluate_initial_population(self.population)
        self.m_f=np.full(5,.5); self.m_cr=np.full(5,.8); self.memory_index=0; self.archive=[]
    def actionable_parameters(self):
        progress=self.budget.total_fes/max(1,self.max_fes)
        return {"F":float(.7 if progress<.25 else np.mean(self.m_f)),"CR":float(.7 if progress<.25 else np.mean(self.m_cr)),"pbest_rate":float(.25-(.25-.11)*progress),"archive_rate":1.0}


class OPAEngine(BaseEngine):
    algorithm_name="OPA"
    canonical_status="research_adapter_verify_against_selected_OPA_paper"
    def _initialize_algorithm(self):
        self.initial_population_size=max(20,8*self.dim); self.population=self.rng.uniform(self.lb,self.ub,(self.initial_population_size,self.dim)); self.fitness=self._evaluate_initial_population(self.population)
    def actionable_parameters(self): return {"drive_weight":0.7,"encircle_weight":0.6,"attack_weight":0.8,"explore_prob":0.35}
    def actionable_bounds(self): return {"drive_weight":(0.1,1.5),"encircle_weight":(0.1,1.5),"attack_weight":(0.1,1.5),"explore_prob":(0.05,0.8)}
    def probe_candidates(self,pop,fit,p,rng):
        best=pop[int(np.argmin(fit))]; mean=np.mean(pop,axis=0); out=[]
        for x in pop:
            peer=pop[int(rng.integers(0,len(pop)))]; drive=p["drive_weight"]*(mean-x); encircle=p["encircle_weight"]*rng.random(self.dim)*(best-x); attack=p["attack_weight"]*rng.random(self.dim)*(best-peer)
            explore=rng.normal(0,.03*(self.ub-self.lb)) if rng.random()<p["explore_prob"] else 0.0
            out.append(np.clip(x+drive+encircle+attack+explore,self.lb,self.ub))
        return np.asarray(out)
    def optimizer_generation(self):
        c=self.probe_candidates(self.population,self.fitness,self.current_parameters(),self.rng); vals,n=self._evaluate_optimizer_candidates(c); s=0
        for i in range(n):
            if vals[i]<self.fitness[i]: self.population[i]=c[i]; self.fitness[i]=vals[i]; s+=1
        return s,n


class SBOAEngine(BaseEngine):
    algorithm_name="SBOA"
    canonical_status="research_adapter_verify_against_selected_SBOA_paper"
    def _initialize_algorithm(self):
        self.initial_population_size=max(20,8*self.dim); self.population=self.rng.uniform(self.lb,self.ub,(self.initial_population_size,self.dim)); self.fitness=self._evaluate_initial_population(self.population)
    def actionable_parameters(self): return {"hunt_weight":0.7,"escape_weight":0.5,"explore_prob":0.4,"local_weight":0.6}
    def actionable_bounds(self): return {"hunt_weight":(0.1,1.5),"escape_weight":(0.1,1.5),"explore_prob":(0.05,0.9),"local_weight":(0.1,1.5)}
    def probe_candidates(self,pop,fit,p,rng):
        best=pop[int(np.argmin(fit))]; out=[]
        for x in pop:
            peer=pop[int(rng.integers(0,len(pop)))]; hunt=p["hunt_weight"]*rng.random(self.dim)*(best-x); local=p["local_weight"]*rng.random(self.dim)*(peer-x)
            escape=p["escape_weight"]*rng.normal(0,.04*(self.ub-self.lb)) if rng.random()<p["explore_prob"] else 0.0
            out.append(np.clip(x+hunt+local+escape,self.lb,self.ub))
        return np.asarray(out)
    def optimizer_generation(self):
        c=self.probe_candidates(self.population,self.fitness,self.current_parameters(),self.rng); vals,n=self._evaluate_optimizer_candidates(c); s=0
        for i in range(n):
            if vals[i]<self.fitness[i]: self.population[i]=c[i]; self.fitness[i]=vals[i]; s+=1
        return s,n


ENGINE_REGISTRY={
    "PSO":PSOEngine,"BA":BAEngine,"OPA":OPAEngine,"SBOA":SBOAEngine,"GWO":GWOEngine,
    "L-SHADE":LSHADEEngine,"GSK":GSKEngine,"jSO":jSOEngine,
}



def run_engine(func_obj,dim:int,max_fes:int,algorithm:str,variant:str,oracle_config:OracleConfig,*,run_id:int,function_id:int,seed:int,trajectory_dir=None,seed_mode:str="paired"):
    if algorithm not in ENGINE_REGISTRY:raise KeyError(f"Unsupported algorithm: {algorithm}")
    engine=ENGINE_REGISTRY[algorithm](func_obj,dim,max_fes,variant,oracle_config,run_id=run_id,function_id=function_id,seed=seed,trajectory_dir=trajectory_dir,seed_mode=seed_mode)
    result=engine.optimize();result["canonical_status"]=engine.canonical_status;return result
