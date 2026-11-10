from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
INP=ROOT/'ORIGINAL_FORMAL_OUTPUTS'
OUT=ROOT
SD=OUT/'SOURCE_DATA'
SD.mkdir(parents=True,exist_ok=True)

def read(p): return pd.read_csv(INP/p)
def write(d,name):
    temp=SD/(name+'.tmp')
    d.to_csv(temp,index=False,float_format='%.17g')
    temp.replace(SD/name)

write(read('FORMAL_SOURCE_DATA/Fig1_analytic_local_vs_full.csv'),'M1_exact_curves.csv')
write(read('FORMAL_SOURCE_DATA/Fig2_certified_one_slot_reversal.csv'),'M2_exact_candidates.csv')
write(pd.DataFrame({'conclusion':['Relative decision-risk ranking','Both replacement-risk signs','Information rankings and risk signs'],
                    'n_per_row':[33700,327258,987818],
                    'epsilon_numerator':[67,1,33], 'epsilon_denominator':[6450,300,17200],
                    'simultaneous_confidence':[.95]*3,'state_rows':[12]*3}), 'S1_certificate_tiers.csv')

A=read('SUPPORTING_SOURCE_R537/R537_INPUT_R533_924.csv')
assert len(A)==924
scale={'balanced':1.,'miss3':3.,'falsealarm3':3.,'reject035':1.}
A['maximum_contract_loss']=A.contract.map(scale)
A['normalized_difference']=A.matched_minus_full/A.maximum_contract_loss
A['magnitude_class']=np.select([A.normalized_difference<-.01,A.normalized_difference>.01],['Improvement >0.01','Worsening >0.01'],default='Absolute difference ≤0.01')
assert (A.matched_minus_full < -1e-12).sum()==846
assert A.magnitude_class.value_counts().to_dict()=={'Improvement >0.01':715,'Absolute difference ≤0.01':186,'Worsening >0.01':23}
write(A,'S2_all_924_budget_contrasts.csv')
freq=A.groupby(['source','magnitude_class']).size().unstack(fill_value=0).reset_index()
write(freq,'S2_empirical_magnitude_counts.csv')

# Eight native source/protocol strata are the pre-existing 50%-quota object.
# The full 924-row pooled-over-budget result remains a different object.
K=['source','protocol','unit']
F=read('FORMAL_SOURCE_DATA/Fig3_six_study_eight_protocol_descriptive.csv')
assert A.budget.value_counts().to_dict()=={.25:308,.5:308,.75:308}
F50=A[A.budget.eq(.5)].copy()
assert len(F50)==308 and int(F50.matched_better_than_full.sum())==284
F50['normalized_loss_difference']=F50.matched_minus_full/F50.maximum_contract_loss
write(F50,'M3_50pct_observations.csv')
strata=F50.groupby(K).agg(n=('matched_minus_full','size'),
  raw_loss_mean=('matched_minus_full','mean'),
  normalized_loss_mean=('normalized_loss_difference','mean'),
  better_than_complete=('matched_better_than_full','sum'),
  better_than_keep=('matched_better_than_no_swap','sum'),
  complete_better_than_keep=('full_better_than_no_swap','sum')).reset_index()
assert len(strata)==8 and int(strata.n.sum())==308 and int(strata.better_than_complete.sum())==284
assert np.allclose(strata.raw_loss_mean.to_numpy(),
     F.set_index(K).reindex(pd.MultiIndex.from_frame(strata[K])).risk_difference.to_numpy(),
     rtol=0,atol=1e-12)
write(strata,'M3_50pct_strata.csv')

write(read('FORMAL_SOURCE_DATA/Fig4_gas_physical_resource_risk.csv'),'M4_six_physical_policies.csv')
G=read('SUPPORTING_SOURCE_R537/R537_gas_decision_sign_transfer.csv')
sign=G.groupby(['level','test_batch','method'])[['N','deployed','false_harm','beneficial','missed_benefit']].sum().reset_index()
assert len(sign)==32
sign['fraction_predicted_beneficial_but_harmful']=sign.false_harm/sign.deployed
assert (sign.deployed>0).all()
write(sign,'M4_all_batch_sign_transfers.csv')

P=read('FORMAL_EEG_R545/R545_reconciled_subject_risk_15264.csv')
W=P.pivot(index=['processing','contract','k','subject','budget'],columns='policy',values='risk').reset_index()
assert W.groupby(['processing','contract','k','subject']).size().eq(3).all()
W=W.groupby(['processing','contract','k','subject'])[['target','full','local']].mean().reset_index()
W['target_minus_full']=W.target-W.full
W['target_minus_local']=W.target-W.local
fold=P[['subject','fold']].drop_duplicates()
assert len(fold)==106 and not fold.subject.duplicated().any()
W=W.merge(fold,on='subject',validate='many_to_one')
primary=W[(W.processing=='calibrated')&(W.contract=='false_alarm_T2_x3')&(W.k==5)].copy()
assert len(primary)==106
write(primary,'M5_all_106_paired_participants.csv')
Q=read('FORMAL_SOURCE_DATA/Fig5_EEG_full_Holm32_family.csv')
assert len(Q)==32
for r in Q.itertuples():
    v=W[(W.processing==r.processing)&(W.contract==r.contract)&(W.k==r.k)]['target_minus_'+r.reference]
    assert abs(v.mean()-r.mean_target_minus_ref)<1e-12
    assert int((v< -1e-12).sum())==r.subjects_better
    assert int((v>1e-12).sum())==r.subjects_worse
write(Q[Q.contract.eq('false_alarm_T2_x3')],'M5_all_false_alarm_conditions.csv')
write(Q,'S6_complete_32_family.csv')
E=['FC5','FC3','FC1','C5','C3','C1','CP5','CP3']
R=['FC6','FC4','FC2','C6','C4','C2','CP6','CP4']
write(pd.DataFrame({'pair':np.arange(8),'original_electrode':E,'alternative_electrode':R}),'M5_eight_physical_pairs.csv')

write(read('SUPPORTING_SOURCE_R538/R538_official_three_tasks_per_subject.csv'),'S3_UCI_all_paired_participants.csv')
write(read('SUPPORTING_SOURCE_R538/R538_official_three_tasks_subject_bootstrap.csv'),'S3_UCI_stored_intervals.csv')
C=read('SUPPORTING_SOURCE_R538/R538_gas_all_20_concentration_three_policy.csv')
assert len(C)==240 and not C.duplicated(['gases','loss','mode']).any()
con=[]
for row in C.to_dict('records'):
    for ref in ['full','local']:
        con.append({**row,'reference':ref,'maximum_contract_loss':scale[row['loss']],
                    'normalized_difference':(row['target']-row[ref])/scale[row['loss']]})
write(pd.DataFrame(con),'S4_all_concentration_contrasts.csv')
T=read('SUPPORTING_SOURCE_R538/R538_prequential_four_folds_one_physical_slot.csv')
assert len(T)==2560 and not T.duplicated(['split','gas_triplet','contract','policy']).any()
T['maximum_contract_loss']=T.contract.map(scale)
T['normalized_future_change']=T.future_change/T.maximum_contract_loss
T['test_batch']=T.split.str.extract(r'test(\d+)').astype(int)
write(T,'S5_all_one_slot_temporal_choices.csv')
use=T.groupby(['contract','test_batch','policy']).agg(n=('replaced','size'),replaced_count=('replaced','sum')).reset_index()
use['replacement_fraction']=use.replaced_count/use.n
write(use,'S5_observed_replacement_frequencies.csv')

M=read('FORMAL_EEG_R545/R545_reconciled_policy_720.csv')
assert len(M)==720
config=[]
for key,g in M.groupby(['processing','budget','policy']):
    assert len(g)==40
    for j in range(8):
        n=sum((int(v)>>j)&1 for v in g.selected_mask)
        config.append(dict(zip(['processing','budget','policy'],key))|{'pair':j,'original_electrode':E[j], 'alternative_electrode':R[j], 'n_decisions':40,'n_alternative':n,'fraction_alternative':n/40})
write(M,'S7_all_720_selected_masks.csv')
write(pd.DataFrame(config),'S7_observed_pair_selection_frequencies.csv')
write(read('SUPPORTING_SOURCE_R540/R540_EEG_16_electrode_information_metrics.csv'),'S8_fixed_training_electrode_scores.csv')
write(read('SUPPORTING_SOURCE_R540/R540_EEG_secondary_5fold_information.csv'),'S8_five_fold_electrode_scores.csv')
write(read('SUPPORTING_SOURCE_R542/R542_subject_level_bootstrap_Holm_56.csv'),'S9_all_56_unequal_search_contrasts.csv')
V=read('SUPPORTING_SOURCE_R542/R542_validation_label_budget_paired.csv')
assert len(V)==7320
escale={'balanced':1.,'miss_T2_x3':3.,'false_alarm_T2_x3':3.,'reject035':1.}
V['maximum_contract_loss']=V.contract.map(escale)
V['normalized_difference_full']=V.diff_vs_full_top10_val/V.maximum_contract_loss
V['normalized_difference_local']=V.diff_vs_local_top10_val/V.maximum_contract_loss
V['label_availability']=V.n_validation_subjects.map(lambda n:'All (21–22)' if n>10 else str(n))
write(V,'S10_all_7320_frozen_label_subset_comparisons.csv')

parts=[]
for filename,key in [('SOURCE_REBUILT_R546/older4/R529_five_settings_1008_tasks.csv','dataset'),
                     ('SOURCE_REBUILT_R546/har/R530_UCI_HAR_780_真实通道任务.csv','source'),
                     ('SOURCE_REBUILT_R546/gas/R531_gas_800_tasks.csv','source')]:
    native=read(filename)
    z=pd.DataFrame({'source':native[key].replace({'UCI_GAS270':'GasDrift'}),
        'native_row':np.arange(1,len(native)+1),'train_local_r2':native.train_r2,
        'train_complete_delta':native.train_Delta,'test_local_r2':native.test_r2,
        'test_complete_delta':native.test_Delta})
    parts.append(z)
landscape=pd.concat(parts,ignore_index=True)
assert len(landscape)==2588 and landscape.source.nunique()==6
assert ((landscape.train_complete_delta-landscape.train_local_r2)>-1e-8).all()
write(landscape,'S11_all_2588_information_scores.csv')

write(read('SUPPORTING_SOURCE_R537/R537_gas_prequential_resource_efficiency.csv'),
      'S12_all_28_unit_time_resource_summaries.csv')
BP=P.pivot(index=['processing','contract','k','budget','subject'],columns='policy',values='risk').reset_index()
BP['maximum_contract_loss']=BP.contract.map(escale)
BP['normalized_difference_full']=(BP.target-BP.full)/BP.maximum_contract_loss
BP['normalized_difference_local']=(BP.target-BP.local)/BP.maximum_contract_loss
assert len(BP)==5088 and BP.groupby(['processing','contract','k','budget']).subject.nunique().eq(106).all()
write(BP,'S13_all_budget_specific_106_person_pairs.csv')

# Native visual structures: exact exports, empirical pairing/counts only.
from fractions import Fraction as F
from math import sqrt
U=[[F(1,3) for j in range(3)] for i in range(3)]
B=[[F(int(i==j))-F(1,3) for j in range(3)] for i in range(3)]
channels={
 ('A','P'):[[U[i][j]+F(1,10)*B[i][j] for j in range(3)] for i in range(3)],
 ('A','Q'):[[U[i][j]-(F(1,10)-F(1,215))*B[i][j] for j in range(3)] for i in range(3)],
 ('B','P'):U,
 ('B','Q'):[[F(103,300),F(97,300),F(1,3)],
             [F(97,300),F(103,300),F(1,3)],
             [F(1,3),F(1,3),F(1,3)]]}
assert F(1,10)*(4+F(1,10))/(4+3*F(1,10))==F(1,10)-F(1,215)
rows=[]
for (candidate,experiment),matrix in channels.items():
 for i,row in enumerate(matrix):
  assert sum(row)==1 and min(row)>=0
  rows.append(dict(candidate=candidate,experiment=experiment,state=i+1,
      p1=float(row[0]),p2=float(row[1]),p3=float(row[2]),
      exact_p1=str(row[0]),exact_p2=str(row[1]),exact_p3=str(row[2]),
      barycentric_x=float(row[1])+float(row[2])/2,
      barycentric_y=sqrt(3)*float(row[2])/2))
def exact_classification_risk(matrix):
 return F(1)-sum(max(matrix[i][j] for i in range(3)) for j in range(3))/3
assert exact_classification_risk(channels['A','Q'])-exact_classification_risk(channels['A','P'])==F(3,86)
assert exact_classification_risk(channels['B','Q'])-exact_classification_risk(channels['B','P'])==-F(1,150)
write(pd.DataFrame(rows),'M2_exact_channel_rows.csv')

F50['normalized_keep_difference']=F50.matched_vs_no_swap/F50.maximum_contract_loss
F50['comparison_id']=F50[['source','protocol','unit','split','contract']].astype(str).agg('|'.join,axis=1)
assert F50.comparison_id.is_unique
def direction(x):
 return np.where(x < -1e-12,'Lower',np.where(x > 1e-12,'Higher','Tie'))
F50['versus_complete_direction']=direction(F50.matched_minus_full)
F50['versus_keep_direction']=direction(F50.matched_vs_no_swap)
assert (F50.versus_complete_direction=='Lower').sum()==284
assert (F50.versus_complete_direction.eq('Lower')==F50.matched_better_than_full).all()
assert (F50.versus_keep_direction.eq('Lower')==F50.matched_better_than_no_swap).all()
write(F50,'M3_paired_relative_and_retention.csv')
joint=F50.groupby(['versus_complete_direction','versus_keep_direction']).size().reset_index(name='n')
assert joint.n.sum()==308
write(joint,'M3_joint_outcome_counts.csv')

G['no_higher_future_loss']=G.deployed-G.false_harm
G['numerical_ties_among_selected']=G.deployed-G.false_harm-G.beneficial
assert (G[['deployed','false_harm','no_higher_future_loss','numerical_ties_among_selected']]>=0).all().all()
write(G,'S14_full_contract_batch_sign_counts.csv')
GF=G.groupby(['level','method'],as_index=False)[['deployed','false_harm','no_higher_future_loss']].sum()
write(GF,'M4_actual_sign_transfer_flows.csv')
GC=G.groupby(['level','contract','method'],as_index=False)[['deployed','false_harm','no_higher_future_loss']].sum()
write(GC,'S14_contract_sign_transfer_flows.csv')

W['maximum_contract_loss']=W.contract.map(escale)
W['normalized_difference_full']=W.target_minus_full/W.maximum_contract_loss
W['normalized_difference_local']=W.target_minus_local/W.maximum_contract_loss
assert len(W)==1696
write(W,'EEG_primary_all_participant_budget_averages.csv')
example=M[(M.processing=='calibrated') & (M.fold==0) & (M.budget==4) &
          (M.contract=='false_alarm_T2_x3') & (M.k==5) & (M.policy=='target')]
assert len(example)==1
mask=int(example.selected_mask.iloc[0]);assert mask.bit_count()==4
write(pd.DataFrame([dict(pair=j,original_electrode=E[j],alternative_electrode=R[j],
 selected_alternative=(mask>>j)&1,selected_electrode=R[j] if (mask>>j)&1 else E[j],
 selected_mask=mask,fold=0,budget=4,processing='calibrated',contract='false_alarm_T2_x3',k=5)
 for j in range(8)]),'M5_example_four_swap_design.csv')
configs=M.groupby(['processing','budget','policy','selected_mask']).size().reset_index(name='n_choices')
assert configs.n_choices.sum()==720 and configs.selected_mask.nunique()==93
write(configs,'S7_observed_joint_configuration_counts.csv')

H=read('SUPPORTING_SOURCE_R542/R542_equal_supervision_policy_subject_risks.csv')
H=H.groupby(['processing','contract','subject','policy']).risk.mean().unstack('policy').reset_index()
historical=[]
HQ=read('SUPPORTING_SOURCE_R542/R542_subject_level_bootstrap_Holm_56.csv')
for q in HQ.itertuples():
 z=H[(H.processing==q.processing)&(H.contract==q.contract)]
 values=z.target_all_val-z[q.reference]
 assert len(values)==106 and abs(values.mean()-q.mean_target_minus_ref)<1e-12
 for sid,value in zip(z.subject,values):
  historical.append(dict(processing=q.processing,contract=q.contract,reference=q.reference,
                         subject=sid,loss_difference=value))
assert len(historical)==5936
write(pd.DataFrame(historical),'EEG_secondary_all_paired_participant_differences.csv')
landscape['complete_minus_pairwise']=landscape.train_complete_delta-landscape.train_local_r2
write(landscape,'S11_information_landscape_and_gaps.csv')

membership=[]
for subset in [(1,2),(1,3),(2,3),(1,2,3)]:
 for state in [1,2,3]:
  membership.append(dict(restriction='-'.join(map(str,subset)),state=state,
       included=state in subset,kernel_scope='common_all_states' if len(subset)==3 else 'specific_to_state_pair'))
write(pd.DataFrame(membership),'M1_state_restriction_membership.csv')
false_alarm=W[W.contract.eq('false_alarm_T2_x3')].copy()
assert len(false_alarm)==424 and false_alarm.groupby(['processing','k']).subject.nunique().eq(106).all()
write(false_alarm,'M5_all_false_alarm_participant_pairs.csv')

(OUT/'PLOTTING_CODE/PALETTE_SEMANTICS.json').write_text(json.dumps({
 'palette':['#484878','#7884B4','#B4C0E4','#E4E4F0','#E4CCD8','#F0C0CC'],
 'comparison_methods':{'Complete':'#484878','Pairwise':'#7884B4','Goal-matched':'#F0C0CC'},
 'blue_purple_ramp':['#E4E4F0','#B4C0E4','#7884B4','#484878'],
 'rose_ramp':['#E4E4F0','#E4CCD8','#F0C0CC','#D38BA3','#A65C79'],
 'paired_direction_patterns':{'lower_both':'#484878','lower_x_higher_y':'#D38BA3','higher_x_lower_y':'#B4C0E4','higher_both':'#A65C79','at_least_one_numerical_tie':'#767676'},
 'empirical_signed_contrasts':{'lower':'blue-purple, strength follows actual negative loss difference','higher':'rose, strength follows actual positive loss difference','zero':'#E4E4F0','scale':'one symmetric data-derived value range per figure; cube-root saturation for color only, linear positions unchanged'},
 'continuous_color_roles':{'M1':'existing t coordinate between exact curves, not uncertainty','M2':'p3 coordinate within exact row convex hulls, not a response/decision surface','M4_S8':'blue-purple no-higher and rose higher outcomes; intensity equals the actual within-rule outcome fraction on a common0–1 scale','S2':'existing nonnegative complete-minus-pairwise score gap','empirical_ridges':'actual loss difference (vertical in S3, horizontal in the other silhouettes); unchanged256-coordinate density trace represented with32vector color strips'},
 'time_order':{'7':'#E4E4F0','8':'#B4C0E4','9':'#7884B4','10':'#484878'},
 'redundancy':'Direct labels, marker shapes, line types and source/pair ordering; color is never the sole code.',
 'schematics':'Illustrations are mathematical/physical selection structures, not anatomical maps or causal networks.',
 'density_layout':'Gaussian display kernel, h=0.9*min(sampleSD,IQR/1.34)*n**(-0.2), with positive finite fallback; 256 coordinates clipped to observed range. No shape for n<8 or degenerate values. All observed values remain plotted; no inferential density, CI or fitted response.'
 },indent=2))

source_map=[]
for p in sorted(SD.glob('*.csv')):
    source_map.append({'file':p.name,'rows':len(pd.read_csv(p)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(SD/'SOURCE_DATA_README.md').write_text('# Figure source data\n\nAll figure values come from the preserved formal outputs or exact mathematical definitions. Preparation is restricted to exporting, pairing, ordering, empirical counts, unit conversions and reproducible plotting-density layouts. No model, decoder, test, bootstrap, candidate search, posterior or response surface is newly estimated.\n\nCurrent Figure 3 uses all 308 original records at the fixed 50% quota, with 284 lower-loss comparisons versus complete-information selection. All three quotas remain distinct in Supplementary Figure 3 (924 rows). Active Methods, Results and Supplementary Table 3 consistently describe the 50% object. The superseded budget-average wording in archived source documents is historical and is explicitly overridden by CURRENT_SOURCE_LOCK.md.\n\nSigned empirical effects are goal-matched minus the named comparator; negative is favorable. Figure 2 instead gives exact replacement-versus-retention classification-risk changes. Figure 4 and Supplementary Figures 6–8 use retention-relative chemical losses. Supplementary Figure 4 is complete-information minus pairwise-information selection. These quantities are not interchangeable.\n\nThe six-source distribution/budget displays, composition grid, one-slot temporal choices, validation-label sensitivity and budget-specific participant plots divide losses by the contract maximum (one or three). Primary EEG inference and the displayed primary/secondary intervals retain original asymmetric-loss units. Every axis/caption states its scale. EEG participant means average the same three budgets within people as the corresponding formal analysis; budget-specific distributions retain each budget separately.\n\nDensity silhouettes are Gaussian display layouts, peak-scaled and clipped to the observed range. For n>=8 and nondegenerate values, bandwidth is 0.9*min(sampleSD,IQR/1.34)*n**(-0.2), with a positive finite fallback to the nonzero scale and a numerical floor of observed-range/1000. For smaller or degenerate groups every point remains plotted without an invented silhouette. DENSITY_LAYOUTS contains the exact coordinates, bandwidth and group n emitted by the plotting code. These are empirical display shapes, not new inferential densities or intervals.\n\nConfiguration and outcome ribbons preserve actual empirical counts and matched membership. Binary configuration ribbons follow the eight positions in the same selected electrode mask; they are not time transitions or neural connections. Rule-specific chemical selections can repeat the same underlying candidate. No-higher future loss includes favorable and numerical-tie outcomes.\n\nSome table filenames retain their original stable source-object identifiers. Final figure numbers and their full data/code/formal-source dependencies are defined by PLOTTING_CODE/FIGURE_SOURCE_MAP.json and UPDATED_MAPPING.csv. SOURCE_DATA_INDEX.json indexes every current CSV, including emitted density layouts. SOURCE_DATA serves the figures; ORIGINAL_FORMAL_OUTPUTS preserves all obtained formal result objects and their original version provenance.\n')
(SD/'SOURCE_DATA_INDEX.json').write_text(json.dumps(source_map,indent=2))
with (SD/'SOURCE_DATA_README.md').open('a') as f:f.write('\nV4 native additions are the exact state/restriction membership from the formal definition and all424existing false-alarm participant-setting pairs (848reference values). M1/M2 gradients are coordinate encodings of t and p3, not newly analyzed response surfaces. Ridge hues encode actual signed x values with a common symmetric range and cube-root color-saturation transform; coordinate positions remain linear. Categories use markers, row position and direct labels. The original six anchors are expanded only within their blue-purple/rose families.\n')
print('Prepared',len(source_map),'source tables; no fitting, resampling or tests executed.')
