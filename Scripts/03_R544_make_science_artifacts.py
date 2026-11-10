from pathlib import Path
import io,zipfile,json,hashlib,datetime
import numpy as np,pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/mnt/data/pa_r544_work')
A=pd.read_csv(O/'R544_fair_candidate_subject_bootstrap_Holm.csv')
A['Holm_p_32_all_settings']=multipletests(A.one_sided_t_p,method='holm')[1]
subject=pd.read_csv(O/'R544_equal_candidate_heldout_subject_risks.csv')
w=subject.pivot(index=['processing','contract','k','subject','budget'],columns='policy',values='risk').reset_index()
worsening=[]
for (pr,co,k),g in w.groupby(['processing','contract','k']):
 q=g.groupby('subject')[['target','full','local']].mean()
 for ref in ('full','local'):
  d=q.target-q[ref]
  worsening.append({'processing':pr,'contract':co,'k':k,'reference':ref,'p_worsening_one_sided':ttest_1samp(d,0,alternative='greater').pvalue})
W=pd.DataFrame(worsening); W['Holm_p_worsening_32']=multipletests(W.p_worsening_one_sided,method='holm')[1]
A=A.merge(W,on=['processing','contract','k','reference'],validate='one_to_one')
A.to_csv(O/'R544_final_inference_32comparison_families.csv',index=False)

fig,ax=plt.subplots(figsize=(9,5.6))
sub=A.query("processing=='calibrated' and contract=='false_alarm_T2_x3'").sort_values(['k','reference'])
labels=[];means=[];el=[];eh=[]
for _,r in sub.iterrows():
 labels.append(f"Top {int(r.k)} candidates · {r.reference} information")
 imp=-100*r.mean_target_minus_ref;means.append(imp)
 el.append(imp-(-100*r.ci_high));eh.append((-100*r.ci_low)-imp)
inds=np.arange(len(labels))
ax.errorbar(means,inds,xerr=[el,eh],fmt='o',capsize=4)
ax.axvline(0,color='black',lw=1,ls='--')
ax.set_yticks(inds,labels);ax.invert_yaxis()
ax.set_xlabel('Held-out decision-risk improvement (percentage points)')
ax.set_title('Equal labeled data and equal portfolio search budget')
ax.grid(axis='x',alpha=.12)
fig.tight_layout(rect=(0,0.075,1,1))
fig.text(.54,.015,'106 subjects · five subject-disjoint folds · three physical-electrode budgets',ha='center',fontsize=8)
fig.savefig(O/'R544_Fig9_equal_candidate_risk_forest.png',dpi=200);plt.close(fig)

# old source manuscript preserved with targeted corrected material, not an overwritten original
with zipfile.ZipFile('/mnt/data/PaperA_R543_当前完整科研交接_公平候选预算与复现断点_20261103.zip') as parent:
 with zipfile.ZipFile(io.BytesIO(parent.read('00_PARENT/PaperA_R542_COMPLETE_ORIGINAL.zip'))) as z:
  old=z.read('02_R542_SCIENCE/R542_NC_integrated_scientific_working_manuscript_v07.md').decode('utf8')
old=old.replace('**Paper A | R542 | Nature Communications-targeted integrated scientific working manuscript, version 0.7 | 3 November 2026**','**Paper A | R544 | Nature Communications-targeted scientific working manuscript, version 0.8 | 5 November 2026**')
old=old.replace('When all design workflows received the same labeled validation subjects, matching retained a 0.00499 risk reduction against a validation-tuned shortlist of the ten best complete-information candidates (95% participant-paired interval: 0.00088–0.00942; exploratory Holm-adjusted one-sided p=0.0491 across the four decisions).','In a more stringent comparison, all methods ranked the same 28 or 70 equal-budget electrode designs using training data and advanced only their five highest-ranked designs to the identical labeled validation group. For the high-false-alarm task with unlabeled participant calibration, task-based preselection reduced held-out loss by 0.010715 versus complete-deficiency preselection and by 0.009664 versus local-deficiency preselection. Across a 32-comparison Holm family spanning two calibration modes, four decision losses, two shortlist sizes and two information baselines, the respective adjusted one-sided p-values were 0.0128 and 0.0386; no corrected comparison indicated significant deterioration. The same effect did not persist under training-only global normalization.')
marker='### 5.10 Equal labeled-data access preserves the principal EEG advantage'
new_method='''### 4.8 Fully equal candidate-budget comparison (R544)

To separate candidate-search resources from the choice of information objective, each training fold trained the same shrinkage linear discriminant model for every physical-electrode mask of the specified budget (28 designs for two or six swaps; 70 for four swaps). Task-matched, complete-deficiency and local-deficiency methods independently ranked **all** eligible masks using only training data: for the task-matched method, the ranking score was mean training-set subject-balanced decision risk; for the other methods, the score was the corresponding additive information criterion for the swapped physical pairs. Each method passed **exactly five or ten** candidates to the **same** labeled validation group, which chose the lowest-risk member, with deterministic mask tie-breaking. Neither candidate ranking nor validation selection accessed held-out subject labels. The full process was repeated for all five fixed subject-disjoint folds and two calibrated versus training-global-standardized feature treatments. The frozen model-level risk of all 256 masks was separately checked against archived R540 validation and held-out tables to a maximum absolute discrepancy below 5×10^−16.

Within each of the 106 held-out subjects we averaged the paired policy-risk contrast over the three prespecified physical-electrode budgets before inference. For each method contrast, a one-sided subject-paired t-test of improvement was evaluated, and an exploratory Holm adjustment was performed over the full 32 comparisons (two feature treatments × four decision contracts × two shortlist sizes × two information baselines). Paired 95% bootstrap intervals were calculated using 20,000 subject-level resamples. The entire comparison family was kept for both improvement and worsening audits. This is a reanalysis of the same exploratory EEG cohort, not a preregistered confirmatory experiment.

'''
old=old.replace(marker,new_method+marker)
insert='''### 5.11 Equal candidate searches retain a condition-specific gain

The earlier shared-label comparison in Section 5.10 gave the information methods only five or ten shortlisted candidates while the task-matched method could validate all 28 or 70 eligible configurations; it therefore quantified two *workflows*, not solely an information criterion effect. The R544 equal-search reanalysis resolves this distinction. With five candidates per method, participant-calibrated, high-false-alarm matching improved subject-held-out risk by **0.010715** (95% participant-paired bootstrap interval for the improvement: **0.004745–0.016909**) relative to complete-information preselection, and by **0.009664** (**0.003805–0.015984**) relative to local-information preselection. The corresponding Holm-adjusted one-sided p-values across all 32 R544 comparisons were **0.0128** and **0.0386**. Across the 32 comparisons, two supported improvement and none supported worsening after the same Holm family correction. The effect was smaller with ten candidates and was not statistically established with training-only global normalization; calibration is therefore a specific condition under which goal-based preselection produced measurable benefit, not a universal improvement claim. All eight electrode positions were retained throughout; no claim of lower total electrode count or financial savings follows from this experiment.

'''
old=old.replace('## 6. Discussion',insert+'## 6. Discussion')
old=old.replace('Thus equal access to validation labels did not eliminate the observed high-false-alarm benefit, although the more competitive baseline diminished the size of the risk advantage.', 'Thus shared labeled validation alone did not eliminate the observed benefit, although unequal candidate search budgets limit attribution in this R542 comparison; the R544 controlled reanalysis in Section 5.11 supplies the primary fair-search evidence.')
old=old.replace('R542 additionally gave the information-based baselines the same labeled validation subjects and allowed top-five/top-ten information-guided portfolio shortlists to be tuned on those labels. The calibrated high-false-alarm benefit remained, with a smaller but measurable margin against the top-ten shortlist. The comparison isolates shared supervision access while retaining different search-space sizes, which are part of the compared design workflows.', 'R542 gave information-based baselines the same validation labels but retained unequal candidate search budgets. R544 subsequently trained and ranked the same number of candidates for all methods before offering equal shortlist sizes to identical validation groups; under the specified participant-calibrated high-false-alarm task, the advantage remained. These controlled results distinguish a goal-based ranking effect from gains due merely to being permitted more validation candidates.')
old=old.replace('**Figure 8. Advantage under equal labeled validation access.**', '**Figure 8. Shared labeled validation access with unequal candidate search spaces (workflow comparison).**')
old=old.replace('## References (working bibliography, not journal styled)','**Figure 9. Equal candidate-search budgets.** Cross-budget held-out risk improvements of task-based preselection relative to complete and local information preselection, after both methods first rank all 28/70 masks on training data and pass only their five or ten best candidates to the same validation group. Error bars show 95% subject-level paired bootstrap intervals. The calibrated high-false-alarm task is shown; all 32 comparisons and multiplicity adjustments are reported in the accompanying table.\n\n## References (working bibliography, not journal styled)')
# preserve all earlier content including old R542 observations with clarified scope
(O/'R544_NC_scientific_working_manuscript_v08.md').write_text(old,encoding='utf8')

summary={
 'run':'R544','dated':'2026-11-05',
 'parent_full_archive_sha256':hashlib.sha256(Path('/mnt/data/PaperA_R543_当前完整科研交接_公平候选预算与复现断点_20261103.zip').read_bytes()).hexdigest(),
 'EEG_subjects':106, 'eeg_windows':9536, 'folds':5,'feature_preprocessing':['calibrated','global_train_only'],
 'electrode_replace_budgets':[2,4,6],'shortlist_sizes':[5,10],
 'candidate_count_by_budget':{'2':28,'4':70,'6':28},
 'comparison_family':32,'Holm_improvements':int((A.Holm_p_32_all_settings<0.05).sum()),
 'Holm_worsenings':int((A.Holm_p_worsening_32<0.05).sum()),
 'max_archival_validation_test_error':float(pd.read_csv(O/'R544_archive_replay_QA.csv').archive_max_abs_difference.max()),
 'high_false_alarm_calibrated_top5':A.query("processing=='calibrated' and contract=='false_alarm_T2_x3' and k==5")[['reference','mean_target_minus_ref','ci_low','ci_high','Holm_p_32_all_settings','Holm_p_worsening_32']].to_dict('records'),
 'R_feature_level_rebuild_executed_in_current_environment':False,
 'R_math_translation_checked_in_python_against_sklearn':True,
 'R_non_EEG_six_source_raw_pipeline_executed':False,
 'end_to_end_all_raw_source_reproduction_complete':False,
 'NatureCommunications_submitted':False
}
(O/'R544_INDEPENDENT_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print('wrote R544 science, figure and audit',len(old),summary)
