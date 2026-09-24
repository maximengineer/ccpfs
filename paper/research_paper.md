# From Risk Trajectories to Follow-Up Appointments: A Capacity-Constrained Scheduling Framework

---

**Abstract** - Post-discharge follow-up reduces readmissions, yet limited clinic capacity prevents early review for all patients. We present a model-agnostic prescriptive framework that converts any patient-level survival curve S(t) into capacity-feasible follow-up appointments by solving a cost-minimising assignment problem that matches patients to available appointment slots. We validate the framework with four survival models - Cox proportional hazards, gradient-boosted survival analysis (GBM), random survival forest (RSF), and the MOTOR transformer foundation model - showing that the three classical models yield schedules of nearly equal quality and that a weaker foundation-model predictor degrades the schedule gracefully. On 27,641 held-out MIMIC-IV discharge episodes, the specialty-constrained optimal scheduler reduces expected cost by 29% relative to capacity-aware uniform day-14 scheduling and schedules follow-up no later than the readmission for 59% of readmitted patients, compared to 38% under the uniform schedule, while respecting per-specialty daily capacity across four clinical pools. A greedy heuristic comes within 0.4% of the exact optimum and schedules the full test cohort in about 150 ms. The framework code is available at https://github.com/maximengineer/ccpfs.

**Keywords** - follow-up scheduling, hospital readmission, survival analysis, min-cost assignment, capacity constraints, clinical decision support

---

## I. Introduction

Hospital readmissions remain a persistent burden: for heart failure alone, 30-day rates exceed 20% [1]. Outpatient follow-up within 30 days reduces all-cause readmission (pooled RR 0.68, 95% CI 0.60-0.75) [2], but this benefit varies dramatically by condition, with extreme between-study heterogeneity (I^2 = 92.7%) [3]. The operational implication is direct: a fixed interval such as 14 days is too late for patients whose risk peaks in the first week and unnecessarily early for stable patients. When everyone receives the same appointment, scarce capacity is consumed by those who derive little benefit while high-risk patients wait.

The core challenge is that prediction and scheduling have developed in isolation. Guidelines recommend early follow-up [1], but limited capacity makes universal early appointments impractical [4]. Machine learning models can now predict post-discharge risk trajectories from electronic health records (EHRs) [5], and foundation models demonstrate that time-to-event representations transfer across clinical sites [6]. Yet these advances remain disconnected from scheduling: models estimate risk but do not prescribe when patients should be seen [4]. To our knowledge, prior work has not operationalised patient-level readmission survival curves into specialty-constrained discharge follow-up scheduling via an explicit capacity-constrained assignment framework. This gap between prediction and clinical scheduling is the central problem we address.

We argue that follow-up timing should be treated as a capacity-constrained optimisation problem driven by individual risk trajectories. When heterogeneous patients compete for limited early slots, optimal allocation depends not on absolute risk but on the marginal benefit each patient derives from being seen sooner - that is, the reduction in adverse-event probability achieved by scheduling an earlier rather than later appointment. Fixed rules cannot capture this patient-level quantity. This paper makes three contributions:

1. A model-agnostic optimisation formulation that consumes any survival curve S(t) and outputs capacity-feasible appointment days, bridging the prediction-to-prescription gap.
2. An exact min-cost assignment solver and a fast greedy heuristic that reduce expected cost by 29% relative to capacity-aware uniform scheduling under identical clinic capacity, while scheduling follow-up no later than the readmission for 59% of readmitted patients (versus 38%).
3. Evidence that coarse risk stratification yields only 5% improvement over uniform scheduling, establishing that patient-level optimisation - not merely risk awareness - drives scheduling gains.

This paper does not propose a new survival model or claim superiority for any particular risk predictor. The contribution is the prescriptive layer that sits atop any model producing patient-level S(t) curves - converting predictions into decisions.

---

## II. Related Work

Three research threads converge on the problem we address, but none bridges the gap between them.

**Survival prediction for readmission.** Machine learning models now produce patient-level risk trajectories from EHR data. Cox proportional hazards models remain a standard baseline [9], while gradient-boosted survival analysis [5] and random survival forests [10] achieve concordance indices of 0.65-0.72 on general readmission cohorts [7]. More recently, the MOTOR foundation model [6] demonstrated that transformer-based time-to-event representations pretrained on large EHR corpora can transfer to downstream tasks. However, all these models predict risk without prescribing timing - they answer "how likely?" but not "when should we intervene?"

**Scheduling under capacity constraints.** Operations research has extensively studied appointment scheduling in healthcare [11], primarily focusing on operational metrics: minimising patient wait times, reducing no-show impact, and optimising clinic throughput. More broadly, healthcare resource allocation has considered prioritisation under constrained capacity, but these formulations typically treat patients as interchangeable or group them by condition rather than consuming individual clinical risk trajectories as direct input to appointment-day optimisation.

**Clinical follow-up guidelines.** The ACC/AHA recommends follow-up within 7-14 days for heart failure [1], but this is condition-specific and assumes sufficient capacity. The extreme heterogeneity in follow-up benefit across 37 studies (I^2 = 92.7%) [3] indicates that one-size-fits-all intervals are suboptimal, yet no guideline incorporates patient-level risk or capacity constraints.

The gap is clear: survival models produce individual risk curves, scheduling models allocate constrained resources, and clinical guidelines prescribe fixed intervals - but we are not aware of prior work that integrates all three. Our framework fills this gap by using survival predictions as input to an optimisation that respects real clinic capacity and outputs individualised appointment days.

---

## III. Framework Design

Having established that fixed-interval follow-up is poorly matched to heterogeneous post-discharge risk, we now describe the framework that converts predicted survival trajectories into feasible appointment decisions. The novelty lies not in a new survival model or scheduling algorithm, but in connecting them: a cost function translates survival predictions into scheduling decisions, and the resulting assignment problem is solved under realistic capacity constraints. Each component is individually mature; the contribution is their integration into a prescriptive pipeline that neither field has produced alone. Fig. 1 illustrates the architecture.

![Fig. 1: CCPFS pipeline - MIMIC-IV discharge data flows through cohort construction (137,054 patients, 275,022 episodes) and feature extraction (192 features across 5 groups), into four survival models producing patient-level S(t) curves, through isotonic calibration, then into the capacity-constrained min-cost assignment scheduler that outputs feasible per-specialty appointment assignments.](diagram.png)

### A. Data and Cohort

We evaluate on MIMIC-IV v3.1 [8], a publicly available clinical database containing de-identified EHR data from all inpatient hospitalisations (ICU and non-ICU) at Beth Israel Deaconess Medical Center. From this, we extract 275,022 eligible discharge episodes (adults aged 18 or older, length of stay of at least 1 day, discharged home) from 137,054 unique patients. The 30-day all-cause readmission rate is 20.5%. Each episode is assigned to one of four specialty pools based on primary ICD-10 diagnosis codes: general medicine (55.1%), cardiology (19.7%), neurology (13.7%), and surgery (11.5%).

The cohort was pre-split at the patient level - 80% train, 10% validation, 10% test - following the MEDS data standard [12], ensuring no patient appears in multiple splits. All reported results are computed on the held-out test set (27,641 episodes, 5,677 events).

### B. Feature Engineering and Survival Models

We extract 192 clinical features spanning five groups: demographics and admission characteristics (5), comorbidity indicators comprising 12 binary ICD-10 condition flags (heart failure, diabetes, chronic kidney disease (CKD), chronic obstructive pulmonary disease (COPD), hypertension, atrial fibrillation, liver disease, malignancy, depression, obesity, stroke, and acute coronary syndrome) plus diagnosis and procedure counts (14 total), laboratory values and vital signs at discharge covering 28 measurements with last, maximum, and minimum aggregations plus missingness indicators (112), prior healthcare utilisation (3), and derived clinical indicators including lab instability ranges, clinical threshold flags, and interaction terms (58).

The framework requires only patient-specific survival curves S_i(t) over a 30-day horizon; any model producing such curves can serve as input. To validate this claim of independence from any specific predictor, we train and compare four models:

**Table I: Survival Model Comparison (N = 27,641 test episodes)**

| Model | C-index | IBS | Description |
|-------|---------|-----|-------------|
| Cox PH [9] | 0.702 | 0.098 | Semi-parametric baseline via lifelines [13] |
| **GBM** [5] | **0.706** | **0.097** | Gradient-boosted survival (scikit-survival [10]) |
| RSF [10] | 0.698 | 0.098 | Random survival forest ensemble |
| MOTOR+GBM [6] | 0.669 | 0.101 | Foundation model embeddings + GBM |

The concordance index (C-index) measures discriminative ability - the probability that, given two patients, the model correctly identifies which will be readmitted first; 1.0 is perfect, 0.5 is random [14]. The Integrated Brier Score (IBS) evaluates both calibration and discrimination jointly over the full 30-day horizon, where lower values indicate better accuracy. All three classical models achieve near-identical performance (C-index 0.698-0.706), consistent with the 0.65-0.72 range reported for all-condition readmission prediction [7].

The MOTOR-T-Base foundation model (143M parameters, pretrained on 2.57M Stanford Medicine EHRs [6]) was evaluated using frozen 768-dimensional embeddings reduced to 64 components via PCA (93% variance retained). Its lower performance (C-index 0.669) reflects domain shift - Stanford Medicine and Beth Israel differ in coding patterns, patient demographics, and care protocols - and the fact that MOTOR's vocabulary lacks ICD-10-CM diagnosis codes, rendering approximately 68% of MIMIC-IV clinical events invisible to the model. This negative result supports the framework's practical value: even when predictor quality degrades, the scheduling layer remains functional. Purpose-built features outperform general-purpose embeddings for this specific task, but both produce S(t) curves that the scheduler consumes without modification.

We chose GBM for the main scheduling experiments because it achieved the highest C-index while also producing well-separated survival curves across risk groups. Because the optimisation layer consumes only S_i(t), replacing the risk model changes neither the scheduling formulation nor the algorithm; Section V.B shows that Cox PH and RSF curves yield schedules of nearly equal quality, and that MOTOR's weaker curves reduce the catch rate by about 4 percentage points.

### C. Calibration

Since survival curves feed directly into the cost function, calibration is critical. If a model systematically overestimates risk, too many patients compete for early slots; if it underestimates, high-risk patients are deferred when they should not be. We discovered this concretely during development: before calibration, the solver assigned nearly all patients to the first week because raw GBM probabilities were inflated. We apply isotonic regression calibration [15] at horizons 7, 14, 21, and 30 days using the validation set. At each horizon t, the predicted risk 1 - S(t) is recalibrated against observed event rates via a monotone regression, improving the agreement between predicted and observed probabilities before the optimizer consumes them.

### D. Cost Function and Optimal Assignment

For patient i assigned to follow-up on day d, the expected cost is:

    Cost_i(d) = C_EVENT * (1 - S_i(d)) + C_VISIT                    (1)

where 1 - S_i(d) is the cumulative probability of readmission by day d. The event cost C_EVENT = 10,000 is an illustrative value informed by published estimates of preventable 30-day readmission costs, which range from approximately EUR 8,000 to EUR 25,000 depending on condition and healthcare system [16]. The visit cost C_VISIT = 150 represents a standard outpatient follow-up consultation, yielding a 67:1 cost ratio. Crucially, the framework's behaviour depends on this ratio rather than absolute values; since the objective (Equation 2) scales linearly with the ratio, the optimal assignment - which patients get early versus late slots - is invariant to the absolute cost magnitudes. We retain the visit-cost term in Equation (1) for interpretability, as it makes the full per-patient cost explicit. However, since each patient receives exactly one appointment, C_VISIT is constant across assignments and cancels from the objective, so the optimiser minimises only adverse-event exposure:

    Minimise   Sum_i Sum_d  x_{i,d} * C_EVENT * (1 - S_i(d))        (2)
    Subject to:
      Sum_d x_{i,d} = 1                 for all patients i            (3)
      Sum_{i in pool_k} x_{i,d} <= C_k  for all days d, pools k      (4)

Constraint (3) assigns each patient exactly one appointment; (4) enforces per-specialty daily limits. We set each pool's daily capacity proportionally to its share of the test cohort, C_k = ceil(n_k / 30), giving 184 cardiology, 130 neurology, 105 surgery, and 504 general-medicine slots per day (27,690 slots over 30 days for 27,641 patients, with fewer than 30 spare slots in any pool). Capacity is therefore binding but feasible: every patient can be scheduled within their own pool, but not every patient can be scheduled on their optimal day. The framework accepts any per-pool daily limit as input; hospitals would configure these values based on their own clinic schedules. Each pool is managed independently: cardiology slots cannot substitute for neurology slots.

This formulation has the structure of a capacitated assignment problem: each patient must be assigned to exactly one day-slot, and each day-slot has limited capacity. We exploit this structure by reformulating the problem as a min-cost assignment [17]: each (specialty, day) pair with capacity C_k is expanded into C_k individual slots, producing a rectangular cost matrix where rows are patients and columns are slots. Because slots in different pools are disjoint, the problem separates into one assignment per pool, each solved exactly with a shortest-augmenting-path (Jonker-Volgenant) assignment algorithm [17], as implemented in SciPy's linear_sum_assignment, in O(n^3) worst-case time. On a single core of an AMD Ryzen 7 7800X3D, the four pools take 46 s (cardiology, 5,518 patients), 16 s (neurology, 3,877), 8 s (surgery, 3,149), and 15 minutes (general medicine, 15,097), about 16 minutes in total; the global problem without pools (27,641 patients) takes about 87 minutes. The greedy heuristic (Section III.E) schedules the same cohort in about 150 ms. The exact solver is therefore used as an offline benchmark; the greedy method is intended for real-time deployment at discharge.

The critical insight is that optimal allocation follows marginal benefit, not absolute risk. A patient with 30% risk by day 7 rising to 45% by day 21 (flat trajectory, marginal saving: EUR 1,500) benefits less from an early slot than one with 8% risk by day 7 rising to 40% by day 21 (steep trajectory, marginal saving: EUR 3,200). The optimal assignment schedules the steep-trajectory patient earlier despite lower absolute risk, because the cost reduction is more than double. This is why the heterogeneity in follow-up benefit (I^2 = 92.7% [3]) demands patient-level optimisation: only marginal-benefit evaluation can allocate scarce capacity efficiently.

### E. Greedy Heuristic

For real-time deployment at discharge, we implement a greedy heuristic that runs in O(N log N + N * H) time, where N is the number of patients and H is the scheduling horizon: a sort followed by one pass over the days for each patient. Within each specialty pool, the algorithm ranks patients by urgency, defined as their largest single-day hazard S_i(d-1) - S_i(d) over days 1-14, and processes the most urgent first. Each patient is assigned to the lowest-cost day that still has capacity in their pool; because cumulative risk never decreases, this is the earliest day with a free slot. Urgency is a fast proxy for marginal benefit rather than the benefit itself, so the heuristic carries no optimality guarantee. On our test cohort it nevertheless comes within 0.4% of the exact optimum under both global (EUR 1,003 vs EUR 999) and specialty constraints (EUR 1,007 vs EUR 1,003); Section IV.B discusses when this gap can be larger.

---

## IV. Experimental Results

With the framework components in place, we now evaluate whether the optimised scheduling approach delivers meaningful improvements over standard practices. All results use calibrated GBM survival curves on the 27,641 held-out test episodes (5,677 readmission events, 20.5% event rate). Each cost figure is the average per-patient expected cost from Equation (1): the readmission probability at the assigned follow-up day multiplied by C_EVENT, plus the EUR 150 visit cost, which every policy incurs once per patient. Lower values indicate that patients are scheduled closer to their individual risk onset, reducing the period of unmonitored exposure. The catch rate measures the fraction of patients who were actually readmitted and whose readmission occurred on or after their assigned follow-up day - meaning a clinician would have had the opportunity to intervene at a follow-up appointment held no later than the readmission. Catch rate should be interpreted as an opportunity-to-intervene proxy rather than direct evidence that the follow-up appointment would have prevented the subsequent readmission.

### A. Scheduling Policy Comparison

**Table II: Scheduling Policy Comparison (N = 27,641)**

| Policy | Capacity | Avg Cost (EUR) | vs Uniform-14 | Catch Rate |
|--------|:--------:|----------:|-----------:|----------:|
| Guideline (ACC/AHA) | No | 1,992 | +40.1% | 9.8% |
| Uniform day 14 | No | 1,422 | baseline | 37.3% |
| Risk bucket | No | 1,350 | -5.1% | 41.3% |
| Uniform-14 (capacity) | Yes | 1,410 | -0.8% | 37.7% |
| Guideline (capacity) | Yes | 1,408 | -0.9% | 38.9% |
| Greedy (global) | Yes | 1,003 | -29.5% | 59.2% |
| Greedy (specialty) | Yes | 1,007 | -29.2% | 59.0% |
| MinCost (global) | Yes | 999 | -29.7% | 59.1% |
| **MinCost (specialty)** | **Yes** | **1,003** | **-29.5%** | **58.9%** |
| Unconstrained (oracle) | No | 254 | -82.2% | 96.8% |

The capacity-aware baselines start from their intended day (day 14, or the guideline day) and move each patient to the nearest day with a free slot in their own specialty pool. All capacity-aware policies schedule every patient within the available slots.

Cost values are means across all 27,641 test patients. Under the specialty MinCost policy, the 6,461 patients booked in days 1-7 have a mean predicted 30-day readmission risk of 42.8%, five times that of the 9,181 patients booked in days 21-30 (8.6%). An early appointment avoids on average EUR 3,574 of expected event cost per patient compared with a day-30 appointment, against EUR 87 for the late group. Early slots therefore go to the patients whose risk accumulates fastest, which is the framework's intended behaviour.

The comparator policies were selected to represent common non-optimizing alternatives rather than to exhaust all possible hand-tuned rules. Uniform day-14 scheduling reflects a widely used fixed-interval practice; the risk-bucket policy tests whether coarse risk awareness alone is sufficient; the guideline-based policy operationalises condition-specific follow-up rules from published ACC/AHA recommendations [1]. Stronger handcrafted baselines are possible, but they would still lack patient-level marginal-benefit optimisation, which is the mechanism driving the observed gains.

### B. Analysis

Five findings emerge from Table II, each supporting the thesis.

**First, patient-level optimisation dominates.** Comparing only capacity-feasible policies, the specialty MinCost solver (EUR 1,003) reduces cost by 28.9% relative to capacity-aware uniform scheduling (EUR 1,410), while catching 58.9% of readmission events versus 37.7%, 1.6 times as many. This is the most directly comparable benchmark, since both policies operate under identical capacity constraints. The capacity-aware guideline policy (EUR 1,408, 38.9%) performs almost identically to capacity-aware uniform scheduling: once capacity forces both rules to spread patients around their intended day, neither uses patient-level risk to decide who moves earlier.

**Second, risk stratification adds negligible value.** We also evaluated several alternative bucket definitions during development (quartiles, clinical thresholds, condition-specific cutoffs), but they consistently failed to capture patients whose risk rose sharply after discharge despite only moderate 30-day probability. The risk-bucket policy reported here assigns patients to three groups using fixed thresholds on their predicted 30-day readmission probability (at least 30% to day 7, 15-30% to day 14, below 15% to day 30) and achieves only 5.1% improvement over uniform scheduling (EUR 1,350 vs EUR 1,422), raising catch rate from 37.3% to just 41.3%. Knowing which patients are high-risk is insufficient; what matters is trajectory shape and marginal benefit, which coarse group-level stratification cannot capture. Optimising at the individual patient level, not risk awareness alone, drives scheduling gains.

**Third, guideline scheduling performs worst.** The ACC/AHA guideline policy costs 40% more than uniform and catches only 9.8% of events, because it assigns heart failure patients to day 14 but defers all others to day 28 - accumulating risk over the intervening period. Condition-specific rules perform poorly on heterogeneous general-medicine cohorts.

**Fourth, separating specialties costs little when capacity matches demand.** With the same total daily capacity, global MinCost, which lets any patient use any slot, is a lower bound on specialty MinCost, because the per-specialty constraints (4) only restrict the global problem. Keeping the four clinics separate raises expected cost by 0.4% (EUR 1,003 vs EUR 999) and lowers the catch rate from 59.1% to 58.9%. Because each pool's capacity is proportional to its caseload, patients in every pool compete for early slots on similar terms, so little is lost by forbidding cross-pool substitution. We expect this separation cost to grow when capacity is mismatched with demand, for example when one clinic is under-resourced relative to its caseload; the per-pool formulation makes such mismatches visible and quantifiable.

**Fifth, the greedy heuristic is near-optimal on this cohort.** It is 0.4% above the exact optimum under both global (EUR 1,003 vs EUR 999) and specialty constraints (EUR 1,007 vs EUR 1,003), with catch rates within 0.1 percentage points. Because the pools are solved independently, any gap arises within a pool, from ranking patients by peak early hazard rather than by the full cost trade-off the exact solver evaluates. On these calibrated curves the two orderings nearly coincide, but this is not guaranteed: on the synthetic Weibull cohort distributed with the code, whose risk trajectories are steeper and more varied, the same heuristic is 38% above the optimum (EUR 1,077 vs EUR 783 per patient). A deployment should therefore compare the greedy schedule with the exact solver on its own historical data before relying on the heuristic alone.

The unconstrained oracle (EUR 254, 96.8% catch rate) represents the theoretical maximum achievable if every patient could be seen on their individual cost-optimal day, which, because cumulative risk only rises, is day 1 for everyone. The gap between the oracle and the specialty MinCost (EUR 254 vs EUR 1,003) quantifies the price of capacity constraints: under the capacity studied here, even the optimal schedule leaves about four times the oracle's expected cost.

---

## V. Discussion

The results above indicate that individualised scheduling substantially outperforms both uniform and guideline-based approaches on the chosen proxy metrics. Accordingly, the present results should be interpreted as evidence that the framework improves scheduling quality under the chosen objective, rather than definitive evidence of reduced readmission in practice. We now consider what these findings mean in practice, how robust they are, and where the framework falls short.

### A. Clinical Implications

The framework produces directly actionable output: a specific appointment day for each patient, computed at discharge, respecting the clinic's actual daily capacity per specialty. Unlike risk scores that require clinical interpretation, the scheduling decision is concrete and implementable. The marginal-benefit principle means patients with steep risk trajectories - those who benefit most from early appointments - are prioritised over patients with higher but flatter risk profiles. This distinction is clinically meaningful: two patients may both have 30% 30-day risk, but if one's risk accumulates primarily in the first week while the other's is spread evenly, the former should be seen first. No guideline or risk-bucket system captures this.

The greedy heuristic's O(N log N + N * H) complexity enables integration into existing electronic discharge workflows, producing a schedule for the full 27,641-patient test cohort in about 150 ms on a single CPU core. For hospitals where exact solvers are impractical, the greedy approach gave up only 0.4% of optimality on our cohort in exchange for immediate deployability, a trade-off that should be re-checked on each hospital's own data (Section IV.B).

### B. Model-Agnostic Validation

Although GBM is used for the primary scheduling experiments because it achieved the best predictive performance, the scheduling formulation itself only requires calibrated patient-level survival curves and can therefore be applied unchanged to any of the four models tested. The four-model comparison provides evidence that the framework does not depend on a particular risk predictor. Three classical models spanning fundamentally different architectures - semi-parametric (Cox PH), ensemble gradient boosting (GBM), and random forests (RSF) - achieve C-indices within 0.008 of each other. Similar discrimination does not by itself imply similar schedules, however, so we test the scheduling outcomes directly below. The MOTOR foundation model, despite a 0.037 C-index deficit from domain shift, still produces survival curves that the optimiser can schedule over. As prediction models improve - whether through foundation model fine-tuning, larger training corpora, or novel architectures - the scheduling framework should benefit automatically without modification to the scheduling formulation.


To test this beyond prediction metrics, we calibrated each model's test curves on its own validation-set predictions (isotonic regression at 7, 14, 21, and 30 days, as in Section III.C) and ran the specialty-constrained MinCost scheduler on each under the same capacity as Table II:

**Table III: Scheduling Outcomes Across Models (MinCost specialty, N = 27,641)**

| Model | C-index | Avg Cost (EUR) | vs Uniform-14 | Catch Rate |
|-------|---------|---------------:|--------------:|-----------:|
| Cox PH | 0.702 | 1,011 | -28.8% | 58.3% |
| **GBM** | **0.706** | **1,003** | **-29.5%** | **58.9%** |
| RSF | 0.698 | 1,011 | -28.8% | 58.0% |
| MOTOR+GBM | 0.669 | 1,058 | -25.5% | 54.7% |

Expected cost is computed from each model's own survival curves, and so is uniform day-14 scheduling in the "vs Uniform-14" column, so costs measure how much each model believes its schedule improves on the fixed interval rather than providing a common yardstick. The catch rate uses observed readmissions and is directly comparable across models; uniform day-14 scheduling catches 37.3% regardless of the model. The three classical models produce schedules of nearly equal quality: their catch rates lie within 0.9 percentage points (58.0-58.9%) and their predicted cost reductions within 0.7 points. MOTOR's weaker curves cost 4.2 percentage points of catch rate relative to GBM (54.7%), yet its schedule still catches 17 points more readmissions than uniform scheduling. The scheduling layer is thus robust to the choice among comparably accurate predictors and degrades gracefully, not catastrophically, when predictor quality falls; its value remains bounded by the quality of the curves it receives.


### C. Limitations

Several limitations should be noted. First, this is a retrospective evaluation on historical data; we demonstrate that the framework produces better-optimised schedules, but cannot establish a causal link between earlier follow-up and reduced readmissions. The pooled RR of 0.68 from meta-analysis [2] provides external evidence for this mechanism, but a prospective randomised trial would be required for causal proof. Second, the survival model relies on features available at discharge; social determinants of health, medication adherence, and home support systems are not captured in structured EHR data but may influence readmission risk. Third, the framework produces a static schedule at discharge; if a patient's condition deteriorates post-discharge, no mechanism currently adjusts the appointment timing. Fourth, the cost function assumes that adverse-event probability is the appropriate objective; in practice, clinical severity, patient preferences, and appointment type may warrant a richer formulation. Fifth, expected costs are evaluated with the same predicted curves that drive the optimisation, which favours the optimised policies by construction; the catch rate, computed from observed readmissions, is the outcome-based check, and it ranks the policies the same way. Sixth, capacity in our evaluation is sized to the test cohort's own caseload in each pool; real clinics may be tighter, looser, or mismatched with demand. Finally, MIMIC-IV originates from a single academic medical centre (Beth Israel Deaconess); multi-centre validation would strengthen generalisability claims.

### D. Cost and Capacity Sensitivity

Although we use a 67:1 event-to-visit cost ratio in the main analysis, the optimal assignment depends on the ratio rather than absolute values, as noted in Section III.D. The per-specialty capacity limits (184/130/105/504 slots per day for cardiology, neurology, surgery, and general medicine) were set proportionally to each pool's share of the test cohort, C_k = ceil(n_k / 30); hospitals adopting this framework would substitute their own daily clinic availability. Future work should examine how scheduling priorities shift under tighter or more relaxed capacity and whether alternative cost structures (e.g., condition-weighted event costs) produce materially different assignments.

### E. Future Work

Three directions follow naturally. First, fine-tuning the MOTOR foundation model on MIMIC-IV data - rather than using frozen embeddings - may close the performance gap with classical models, testing whether domain adaptation overcomes the vocabulary and distribution mismatch we observed. Second, dynamic rescheduling that incorporates post-discharge data (e.g., patient-reported symptoms, remote monitoring alerts) would enable adaptive scheduling that responds to evolving risk. Third, prospective evaluation in a clinical setting, comparing patient outcomes under optimised versus standard scheduling, would establish whether the theoretical cost reductions translate to actual readmission prevention.

---

## VI. Conclusion

We presented a framework that bridges the gap between risk prediction and clinical scheduling by formulating follow-up appointment timing as an optimisation problem driven by individual risk trajectories under real capacity limits. The extreme heterogeneity in follow-up benefit documented across 37 studies (I^2 = 92.7%) [3] implies that no fixed interval can be optimal for all patients - our results confirm this directly: the specialty-constrained optimal scheduler achieves a 29% cost reduction versus capacity-aware uniform scheduling and schedules follow-up no later than the readmission for 59% of readmitted patients (versus 38%), precisely because it exploits the individual variation that fixed rules ignore. The modest 5% improvement from risk-bucket stratification confirms that per-patient scheduling - not merely sorting patients by risk - is what produces these gains. Validation across four survival models, including a foundation model exhibiting domain-shift degradation, suggests that the scheduling layer is robust to the choice of upstream predictor: as risk models improve, the framework should benefit automatically without modification to the scheduling formulation. The greedy heuristic, within 0.4% of the optimum on our cohort and running in about 150 ms, enables real-time deployment at discharge, making individualised, capacity-feasible scheduling practical for hospital implementation.

---

## References

[1] T. M. Maddox et al., "2024 ACC expert consensus decision pathway for treatment of heart failure with reduced ejection fraction," Journal of the American College of Cardiology, vol. 83, no. 15, pp. 1444-1488, 2024. doi: 10.1016/j.jacc.2023.12.024

[2] I. Balasubramanian, E. B. Andres, and C. Malhotra, "Outpatient follow-up and 30-day readmissions: A systematic review and meta-analysis," JAMA Network Open, vol. 8, no. 11, e2541272, 2025.

[3] D. J. Bilicki and M. J. Reeves, "Outpatient follow-up visits to reduce 30-day all-cause readmissions for heart failure, COPD, myocardial infarction, and stroke: A systematic review and meta-analysis," Preventing Chronic Disease, vol. 21, p. E74, 2024. doi: 10.5888/pcd21.240138

[4] CADTH, "Artificial intelligence for patient flow," CADTH Horizon Scan, vol. 4, no. 4, Apr. 2024.

[5] S. Davis and R. Greiner, "Survival models and longitudinal medical events for hospital readmission forecasting," BMC Health Services Research, vol. 24, no. 1, p. 1394, 2024. doi: 10.1186/s12913-024-11771-w

[6] E. Steinberg, J. Fries, Y. Xu, and N. Shah, "MOTOR: A time-to-event foundation model for structured medical records," in Proceedings of the International Conference on Learning Representations (ICLR), 2024.

[7] P. Pons-Suner et al., "Prediction of 30-day unplanned hospital readmission through survival analysis," Heliyon, vol. 9, no. 10, e20942, 2023.

[8] A. E. W. Johnson, L. Bulgarelli, L. Shen, A. Gayles, A. Shammout, S. Horng, T. J. Pollard, B. Moody, B. Gow, L.-w. H. Lehman, L. A. Celi, and R. G. Mark, "MIMIC-IV, a freely accessible electronic health record dataset," Scientific Data, vol. 10, art. 1, 2023. doi: 10.1038/s41597-022-01899-x

[9] D. R. Cox, "Regression models and life-tables," Journal of the Royal Statistical Society: Series B, vol. 34, no. 2, pp. 187-220, 1972.

[10] S. Polsterl, "scikit-survival: A library for time-to-event analysis built on top of scikit-learn," Journal of Machine Learning Research, vol. 21, no. 212, pp. 1-6, 2020.

[11] A. Ala, F. E. Alsaadi, M. Ahmadi, and S. Mirjalili, "Optimization of an appointment scheduling problem for healthcare systems based on the quality of fairness service using whale optimization algorithm and NSGA-II," Scientific Reports, vol. 11, art. 19816, 2021.

[12] M. McDermott et al., "MEDS: Medical Event Data Standard for structured EHR data," in NeurIPS 2024 Workshop on Learning from Time Series for Health, 2024. [Online]. Available: https://github.com/Medical-Event-Data-Standard

[13] C. Davidson-Pilon, "lifelines: survival analysis in Python," Journal of Open Source Software, vol. 4, no. 40, p. 1317, 2019.

[14] F. E. Harrell, K. L. Lee, and D. B. Mark, "Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors," Statistics in Medicine, vol. 15, no. 4, pp. 361-387, 1996.

[15] B. Van Calster et al., "Calibration: the Achilles heel of predictive analytics," BMC Medicine, vol. 17, art. 230, 2019.

[16] S. F. Jencks, M. V. Williams, and E. A. Coleman, "Rehospitalizations among patients in the Medicare fee-for-service program," New England Journal of Medicine, vol. 360, no. 14, pp. 1418-1428, 2009. doi: 10.1056/NEJMsa0803563

[17] R. Burkard, M. Dell'Amico, and S. Martello, Assignment Problems, revised reprint. Philadelphia, PA: SIAM, 2012.
