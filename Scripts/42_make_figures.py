#!/usr/bin/env python3
"""Rebuild five main and fourteen supplementary scientific figures.

Only existing exact channels, formal results and permitted deterministic
display transformations are used. No model fit, policy search, sampling,
bootstrap, test, new response surface or confidence interval is computed.
"""
import figure_engine as E
from figure_engine import *
from main_figures import M1, M2, M3, M4, M5, sign_flow, harmful_colorbar

def source_label(name):
    return {'MEX':'MEx','UCI_HAR':'UCI HAR'}.get(name,name)

def off(ax):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')












def S1():
    d=data('S1_certificate_tiers.csv');fig=canvas(107)
    a=fig.add_axes([.07,.15,.47,.70]);b=fig.add_axes([.70,.15,.26,.70])
    label(a,'a');label(b,'b');off(a);head(a,'Nested conclusions in the exact construction')
    boxes=[(.02,.02,.96,.90,P['pale']),(.08,.08,.84,.59,P['soft']),(.15,.14,.70,.28,P['rose'])]
    for x,y,w,h,co in boxes:a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.025',facecolor=co,edgecolor=P['dark'],lw=1))
    a.text(.5,.80,'Information ordering and both risk signs',ha='center',fontsize=6.5)
    a.text(.5,.72,'ε < 33/17,200',ha='center',fontsize=6.5)
    a.text(.5,.57,'Both individual replacement-risk signs',ha='center',fontsize=6.5)
    a.text(.5,.49,'ε < 1/300',ha='center',fontsize=6.5)
    a.text(.5,.31,'Relative decision-risk ranking',ha='center',fontsize=6.5)
    a.text(.5,.22,'ε < 67/6,450',ha='center',fontsize=6.5)
    for y,row,col in zip([0,1,2],d.itertuples(),[P['rose'],P['soft'],P['dark']]):
        b.scatter(row.n_per_row,y,s=32,color=col,edgecolors=P['dark'],lw=1,zorder=3)
        b.annotate(f'{row.n_per_row:,}',(row.n_per_row,y),xytext=(9 if y==2 else 0,12),textcoords='offset points',ha='right' if y==2 else 'center',fontsize=7)
    b.set(xscale='log',xlim=(2e4,1.5e6),ylim=(-.4,2.5),xlabel='Independent observations per row')
    b.set_yticks([0,1,2],['Relative','Signs','Ordering + signs']);b.set_xticks([1e5,1e6],['100,000','1,000,000']);b.minorticks_off()
    head(b,'Sufficient counts · 95% simultaneous')
    save(fig,'S1',[a,b],rows=[['a','b']],cols=[])

def S2():
    d=data('S11_information_landscape_and_gaps.csv');fig=canvas(127)
    a=fig.add_axes([.12,.18,.43,.69]);b=fig.add_axes([.76,.18,.20,.69]);label(a,'a');label(b,'b')
    span=d[['train_local_r2','train_complete_delta']].to_numpy().max()*1.04
    a.add_patch(Polygon([[0,0],[span,0],[span,span]],facecolor=P['pale'],edgecolor='none',lw=0,zorder=0))
    a.plot([0,span],[0,span],color=P['grey'],ls='--',lw=1)
    markers=['o','s','^','D','v','P']
    for j,(src,marker) in enumerate(zip(SOURCE,markers)):
        q=d[d.source==src];co=P['rose'] if src=='GasDrift' else P['mid']
        a.scatter(q.train_local_r2,q.train_complete_delta,s=6,marker=marker,c=E.effect_colors(q.complete_minus_pairwise.to_numpy(),points=True),edgecolors='none',alpha=.65,zorder=2)
        cloud(b,q.complete_minus_pairwise,5-j,co,height=.36,point_size=1.3,jitter=.055,marker=marker,tag='S2_'+src)
    a.set(xlim=(-.025*span,span),ylim=(-.025*span,span),xlabel='Pairwise deficiency r₂',ylabel='Complete deficiency Δ')
    a.text(.12,.82,'r₂ ≤ Δ',transform=a.transAxes,fontsize=7)
    head(a,'All 2,588 channel comparisons');head(b,'Complete − pairwise')
    lo=d.complete_minus_pairwise.min();hi=d.complete_minus_pairwise.max()
    b.set(xlim=(lo-.04*(hi-lo),hi*1.08),ylim=(-.5,5.55),xlabel='Deficiency gap')
    b.set_yticks(range(5,-1,-1),[source_label(s)+f' · {sum(d.source==s):,}' for s in SOURCE]);zero(b)
    a.xaxis.set_major_locator(MaxNLocator(4));a.yaxis.set_major_locator(MaxNLocator(4));b.xaxis.set_major_locator(MaxNLocator(3))
    fig.legend(handles=[Line2D([],[],marker=m,ls='',ms=3.5,color=P['dark'],
               mfc=P['soft'],label=source_label(src))
               for src,m in zip(SOURCE,markers)],loc='lower center',bbox_to_anchor=(.54,.028),
               ncol=6,columnspacing=1,handletextpad=.4)
    save(fig,'S2',[a,b],rows=[['a','b']],cols=[])

LOSSCOL={'balanced':P['dark'],'miss3':P['mid'],'falsealarm3':P['pink'],'reject035':P['rose']}
LOSSMARK={'balanced':'o','miss3':'s','falsealarm3':'^','reject035':'D'}

def S3():
    d=data('S2_all_924_budget_contrasts.csv');fig=canvas(183);axes=[]
    gs=fig.add_gridspec(3,2,left=.12,right=.97,bottom=.13,top=.92,wspace=.40,hspace=.52)
    spanlo=min(-.3,d.normalized_difference.min()-.015);spanhi=max(.10,d.normalized_difference.max()+.015)
    for i,src in enumerate(SOURCE):
        ax=fig.add_subplot(gs[i//2,i%2]);axes.append(ax);label(ax,chr(97+i));head(ax,source_label(src))
        q=d[d.source==src]
        for key,z in q.groupby(['protocol','unit','split','contract']):
            z=z.sort_values('budget');assert len(z)==3
            paired_paths(ax,[0,1,2],z.normalized_difference,color=P['mid'],alpha=.14)
        for j,budget in enumerate([.25,.5,.75]):
            v=q[q.budget==budget];den=display_density(v.normalized_difference,f'S3|{src}|{budget}')
            if den is not None:
                yy,dd=den;E.vertical_value_ridge(ax,yy,dd,j+.08,width=.20,zorder=2)
            for contract in LOSS4:
                t=v[v.contract==contract]
                ax.scatter(np.full(len(t),j),t.normalized_difference,s=5,marker=LOSSMARK[contract],c=E.effect_colors(t.normalized_difference.to_numpy(),points=True),edgecolors='none',alpha=.82,zorder=3)
        zero(ax,'y');ax.set(xlim=(-.18,2.42),ylim=(spanlo,spanhi))
        ax.set_xticks([0,1,2],['25','50','75']);ax.set_yticks([-.3,-.2,-.1,0,.1])
        if i%2==0:ax.set_ylabel('Normalized loss difference')
        if i>=4:ax.set_xlabel('Replacement quota (%)')
    fig.legend(handles=[Line2D([],[],marker=LOSSMARK[k],color=P['dark'],mfc='white',mec=P['dark'],ls='',ms=3.5,label=LOSSNAME[k]) for k in LOSS4],
        loc='lower center',bbox_to_anchor=(.54,.025),ncol=4,columnspacing=1)
    save(fig,'S3',axes,rows=[['a','b'],['c','d'],['e','f']],cols=[['a','c','e'],['b','d','f']])

def short_loss(name):
    if 'miss' in name or 'third_state_cost3' in name:return 'Miss ×3'
    if 'false' in name:return 'False alarm ×3'
    if 'reject' in name or 'abstain' in name:return 'Abstention'
    return 'Equal cost'

def S4():
    d=data('S3_UCI_all_paired_participants.csv');q=data('S3_UCI_stored_intervals.csv')
    fig=canvas(145);a=fig.add_axes([.27,.15,.69,.75]);label(a,'a')
    labels=[]
    for j,r in enumerate(q.sort_values(['states','contract']).itertuples()):
        v=d[(d.states==r.states)&(d.contract==r.contract)].sort_values('subject').full_minus_local
        assert len(v)==9
        cloud(a,v,11-j,P['mid'],height=.35,tag=f'S4|{r.states}|{r.contract}',point_size=7,jitter=.075)
        interval(a,r.full_minus_local,11-j-.035,r.ci_lower,r.ci_upper,P['dark'],'D',False,3.7)
        labels.append(r.states+' · '+short_loss(r.contract))
        if j in [3,7]:a.axhline(10.5-j,color=P['light'],lw=1,zorder=0)
    zero(a);a.set(xlim=(min(d.full_minus_local.min(),q.ci_lower.min())-.025,
                       max(d.full_minus_local.max(),q.ci_upper.max())+.025),ylim=(-.55,11.55),
                 xlabel='Complete − pairwise selection: held-out loss difference')
    a.set_yticks(range(11,-1,-1),labels);a.xaxis.set_major_locator(MaxNLocator(5))
    head(a,'Three activity-state sets · nine paired participants per contrast')
    save(fig,'S4',[a],rows=[],cols=[])

def S5():
    E.composition_matrix()

BATCHCOLOR={7:P['pale'],8:P['soft'],9:P['mid'],10:P['dark']}
BATCHMARK={7:'o',8:'s',9:'D',10:'^'}

def S6():
    d=data('S5_all_one_slot_temporal_choices.csv');use=data('S5_observed_replacement_frequencies.csv')
    policies=['keep','local','full','target','val_optional_target','train_and_val_optional','val_target_dominance','val_best_four']
    # Use the formal policy order/names; the exact native labels are asserted.
    policy_labels={'keep':'Keep original','local':'Pairwise ranking','full':'Complete ranking','target':'Goal-matched ranking',
       'val_optional':'Previous-batch optional','validation_optional':'Previous-batch optional',
       'blend_optional':'Training + previous batch','dom_optional':'Previous-batch dominance',
       'dominance_optional':'Previous-batch dominance','four_way_val':'Four-way validation choice',
       'fourway_validation':'Four-way validation choice','val_optional_target':'Previous-batch optional',
       'train_and_val_optional':'Training + previous batch','val_target_dominance':'Previous-batch dominance',
       'val_best_four':'Four-way validation choice'}
    actual=list(d.policy.unique())
    assert len(policies)==8 and set(policies)==set(actual)
    assert all(p in policy_labels for p in policies)
    fig=canvas(238);axes=[]
    gs=fig.add_gridspec(4,2,left=.30,right=.97,bottom=.095,top=.95,wspace=.29,hspace=.62,width_ratios=[1.6,1])
    for i,contract in enumerate(LOSS4):
        a=fig.add_subplot(gs[i,0]);b=fig.add_subplot(gs[i,1]);axes.extend([a,b]);label(a,chr(97+2*i));label(b,chr(98+2*i))
        for j,policy in enumerate(policies):
            y=7-j;v=d[(d.contract==contract)&(d.policy==policy)]
            assert len(v)==80
            cloud(a,v.normalized_future_change,y,P['mid'],height=.52,points=False,tag=f'S6|{contract}|{policy}')
            for ii,batch in enumerate([7,8,9,10]):
                w=v[v.test_batch==batch];assert len(w)==20
                offy=(1.5-ii)*.20
                a.scatter(w.normalized_future_change,y-.07+offy+point_jitter(20,.04),s=3.1,
                          marker=BATCHMARK[batch],color=BATCHCOLOR[batch],edgecolors=P['dark'] if batch==7 else 'none',lw=1 if batch==7 else 0,zorder=3)
                u=use[(use.contract==contract)&(use.policy==policy)&(use.test_batch==batch)].iloc[0]
                b.barh(y+offy,float(u.replacement_fraction)*100,height=.13,color=BATCHCOLOR[batch],edgecolor='none',lw=0,zorder=2)
                b.scatter(float(u.replacement_fraction)*100,y+offy,s=5.5,marker=BATCHMARK[batch],facecolors=BATCHCOLOR[batch],edgecolors=P['dark'],lw=1,zorder=4)
        zero(a);a.set(ylim=(-.5,7.6),xlim=(-.52,.52));b.set(ylim=(-.5,7.6),xlim=(-4,104))
        a.set_yticks(range(7,-1,-1),[policy_labels[p] for p in policies]);b.set_yticks([])
        a.set_xticks([-.4,0,.4]);b.set_xticks([0,50,100])
        head(a,LOSSNAME[contract]+' · future loss');head(b,'Physical replacements')
        if i==3:a.set_xlabel('Normalized loss change versus keep');b.set_xlabel('Cases using one substitution (%)')
    fig.legend(handles=[Line2D([],[],marker=BATCHMARK[b],ls='',mfc=BATCHCOLOR[b],mec=P['dark'],color=P['dark'],ms=3.7,label=f'Batch {b}') for b in [7,8,9,10]],
               loc='lower center',bbox_to_anchor=(.62,.025),ncol=4,columnspacing=1.5)
    save(fig,'S6',axes,rows=[['a','b'],['c','d'],['e','f'],['g','h']],cols=[['a','c','e','g'],['b','d','f','h']])

def S7():
    d=data('S12_all_28_unit_time_resource_summaries.csv');assert len(d)==28
    pol=['full_fixed','target_fixed','target_optional','validation_optional','blend_optional','agreement_optional','oracle_optional']
    names=['Complete · quota','Goal-matched · quota','Goal-matched · ceiling','Previous-batch · ceiling','Blended · ceiling','Agreement · ceiling','Hindsight oracle']
    fig=canvas(152);a=fig.add_axes([.30,.19,.36,.66]);b=fig.add_axes([.78,.19,.19,.66])
    label(a,'a');label(b,'b');head(a,'Held-out loss versus retention');head(b,'Allowed capacity used')
    styles=[('task','all_forward_7to10',.27,P['dark'],'o',False,'Tasks · batches 7–10'),
            ('task','last_batch_10',.09,P['dark'],'o',True,'Tasks · batch 10'),
            ('physical','all_forward_7to10',-.09,P['rose'],'s',False,'Physical · batches 7–10'),
            ('physical','last_batch_10',-.27,P['rose'],'s',True,'Physical · batch 10')]
    for j,policy in enumerate(pol):
        for lev,time,offy,co,m,opened,text in styles:
            r=d[(d.level==lev)&(d.time_scope==time)&(d.policy==policy)].iloc[0]
            a.scatter(r.mean_risk_change,6-j+offy,s=18,marker=m,facecolors='white' if opened else co,edgecolors=co if co==P['dark'] else P['edge'],lw=1,zorder=3)
            b.barh(6-j+offy,r.fraction_capacity_used*100,height=.12,facecolor='white' if opened else co,
                   edgecolor=co if co==P['dark'] else P['edge'],lw=1,zorder=3)
    zero(a);a.set(xlim=(-.042,.006),ylim=(-.6,6.6),xlabel='Mean loss change versus keep');a.set_xticks([-.04,-.02,0]);a.set_yticks(range(6,-1,-1),names)
    b.set(xlim=(-3,103),ylim=(-.6,6.6),xlabel='Used / allowed (%)');b.set_yticks([]);b.set_xticks([0,50,100])
    a.axhline(.5,color=P['light'],lw=1);b.axhline(.5,color=P['light'],lw=1)
    fig.legend(handles=[Line2D([],[],color=co,marker=m,mfc='white' if op else co,mec=co if co==P['dark'] else P['edge'],ls='',ms=3.8,label=txt) for lev,time,offy,co,m,op,txt in styles],
               loc='lower center',bbox_to_anchor=(.62,.035),ncol=2,columnspacing=1.5)
    save(fig,'S7',[a,b],rows=[['a','b']],cols=[])

def S8():
    d=data('S14_contract_sign_transfer_flows.csv');fig=canvas(235);axes=[]
    gs=fig.add_gridspec(4,2,left=.10,right=.94,bottom=.10,top=.94,wspace=.42,hspace=.45)
    for i,contract in enumerate(LOSS4):
        for j,level in enumerate(['task','physical']):
            ax=fig.add_subplot(gs[i,j]);axes.append(ax);label(ax,chr(97+2*i+j))
            sign_flow(ax,d[(d.contract==contract)&(d.level==level)],'S8|'+contract+'|'+level,font=6.1)
            head(ax,LOSSNAME[contract]+(' · tasks' if level=='task' else ' · physical slots'))
    harmful_colorbar(fig,[.30,.048,.46,.009])
    save(fig,'S8',axes,rows=[['a','b'],['c','d'],['e','f'],['g','h']],cols=[['a','c','e','g'],['b','d','f','h']])

REFS=['full_info_only','local_info_only','full_local_val_pick','full_top5_val','local_top5_val','full_top10_val','local_top10_val']
REFNAME={'full_info_only':'Complete only','local_info_only':'Pairwise only','full_local_val_pick':'Complete / pairwise pick',
         'full_top5_val':'Complete top 5','local_top5_val':'Pairwise top 5','full_top10_val':'Complete top 10','local_top10_val':'Pairwise top 10'}

def uncertainty_family(ident,historical=False):
    fig=canvas(234);axes=[]
    gs=fig.add_gridspec(4,2,left=.25,right=.97,bottom=.075,top=.95,wspace=.45,hspace=.65,width_ratios=[1.65,1])
    if historical:
        persons=data('EEG_secondary_all_paired_participant_differences.csv');stats=data('S9_all_56_unequal_search_contrasts.csv')
    else:
        persons=data('EEG_primary_all_participant_budget_averages.csv');stats=data('S6_complete_32_family.csv')
    for i,loss in enumerate(EEGLOSS):
        a=fig.add_subplot(gs[i,0]);b=fig.add_subplot(gs[i,1]);axes.extend([a,b]);label(a,chr(97+i*2));label(b,chr(98+i*2))
        allvalues=[];labels=[]
        if historical:
            for j,ref in enumerate(REFS):
                labels.append(REFNAME[ref]);y=6-j
                for proc,dy,co,mark in [('calibrated',.18,P['dark'],'o'),('global_train_only',-.18,P['rose'],'s')]:
                    v=persons[(persons.processing==proc)&(persons.contract==loss)&(persons.reference==ref)].loss_difference
                    r=stats[(stats.processing==proc)&(stats.contract==loss)&(stats.reference==ref)].iloc[0]
                    assert len(v)==106;allvalues.extend(v)
                    cloud(a,v,y+dy,co,height=.25,side=1 if dy>0 else -1,tag=f'{ident}|{loss}|{ref}|{proc}',point_size=1.6,marker=mark,jitter=.045,point_color=P['dark'] if proc=='calibrated' else P['mid'])
                    interval(b,r.mean_target_minus_ref,y+dy,r.CI_2p5,r.CI_97p5,co,mark,proc!='calibrated',3.4)
            yt=list(range(6,-1,-1));ylim=(-.6,6.6)
            lows=stats[stats.contract==loss].CI_2p5;highs=stats[stats.contract==loss].CI_97p5
        else:
            settings=[('calibrated',5),('calibrated',10),('global_train_only',5),('global_train_only',10)]
            for j,(proc,k) in enumerate(settings):
                labels.append(('Calibrated' if proc=='calibrated' else 'Global')+f' · {k}');y=3-j
                z=persons[(persons.processing==proc)&(persons.contract==loss)&(persons.k==k)]
                for ref,dy,mark in [('full',.18,'o'),('local',-.18,'s')]:
                    v=z['target_minus_'+ref];allvalues.extend(v);assert len(v)==106
                    r=stats[(stats.processing==proc)&(stats.contract==loss)&(stats.k==k)&(stats.reference==ref)].iloc[0]
                    cloud(a,v,y+dy,COL[ref],height=.26,side=1 if dy>0 else -1,tag=f'{ident}|{loss}|{proc}|{k}|{ref}',point_size=1.6,marker=mark,jitter=.045)
                    interval(b,r.mean_target_minus_ref,y+dy,r.ci_low,r.ci_high,COL[ref],mark,ref=='local',3.4)
            yt=list(range(3,-1,-1));ylim=(-.6,3.6)
            lows=stats[stats.contract==loss].ci_low;highs=stats[stats.contract==loss].ci_high
        vl,vh=min(allvalues),max(allvalues);pad=max((vh-vl)*.055,1e-5)
        a.set(xlim=(vl-pad,vh+pad),ylim=ylim);a.set_yticks(yt,labels)
        cl,ch=lows.min(),highs.max();cp=max((ch-cl)*.12,1e-6)
        b.set(xlim=(cl-cp,ch+cp),ylim=ylim);b.set_yticks([])
        zero(a);zero(b);a.xaxis.set_major_locator(MaxNLocator(4));b.xaxis.set_major_locator(MaxNLocator(3))
        head(a,LOSSNAME[loss]+' · participant differences');head(b,'Mean · 95% paired interval')
        if i==3:a.set_xlabel('Individual loss difference');b.set_xlabel('Mean loss difference')
    handles=([Line2D([],[],marker='o',color=P['dark'],ls='',label='Participant calibration'),
              Line2D([],[],marker='s',color=P['dark'],mfc='white',ls='',label='Training-global scaling')] if historical else
             [Line2D([],[],marker='o',color=COL['full'],ls='',label='Versus complete'),
              Line2D([],[],marker='s',color=COL['local'],mfc='white',ls='',label='Versus pairwise')])
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.61,.018),ncol=2,columnspacing=1.8)
    save(fig,ident,axes,rows=[['a','b'],['c','d'],['e','f'],['g','h']],cols=[['a','c','e','g'],['b','d','f','h']])

def S9():uncertainty_family('S9',False)

def S10():
    d=data('S7_observed_joint_configuration_counts.csv');pairs=data('M5_eight_physical_pairs.csv')
    fig=canvas(226);axes=[]
    gs=fig.add_gridspec(3,2,left=.14,right=.97,bottom=.12,top=.93,wspace=.40,hspace=.37)
    for i,budget in enumerate([2,4,6]):
        for j,proc in enumerate(['calibrated','global_train_only']):
            ax=fig.add_subplot(gs[i,j]);axes.append(ax);label(ax,chr(97+2*i+j));off(ax)
            head(ax,('Participant calibration' if proc=='calibrated' else 'Training-global')+f' · {budget} swaps')
            for idx,policy in enumerate(['full','local','target']):
                q=d[(d.processing==proc)&(d.budget==budget)&(d.policy==policy)]
                assert q.n_choices.sum()==40
                rec=[(tuple((int(r.selected_mask)>>k)&1 for k in range(8)),r.n_choices,policy) for r in q.itertuples()]
                bottom=.70-.30*idx
                alluvial(ax,[str(k+1) for k in range(8)],[[0,1] for k in range(8)],rec,
                    tag=f'S10|{proc}|{budget}|{policy}',bounds=(0,bottom,.98,.22),colors=COL,
                    node_labels=False,node_width=.012,gap=.12,alpha=.55,frequency_opacity=True)
                ax.text(-.025,bottom+.11,{'full':'Complete','local':'Pairwise','target':'Goal-matched'}[policy],ha='right',va='center',fontsize=6.3)
            for k in range(8):ax.text(.98*(.05+.9*k/7),.985,str(k+1),ha='center',fontsize=6.3)
    fig.text(.55,.065,'Upper node: original · lower node: alternative · 40 selected configurations per method and panel',ha='center',fontsize=6.5)
    for j,r in enumerate(pairs.itertuples()):
        fig.text(.12+j*.107,.030,f'{r.pair+1}: {r.original_electrode}/{r.alternative_electrode}',ha='center',fontsize=6.0)
    save(fig,'S10',axes,rows=[['a','b'],['c','d'],['e','f']],cols=[['a','c','e'],['b','d','f']])

def S11():E.electrode_scores()

def S12():
    d=data('S13_all_budget_specific_106_person_pairs.csv');fig=canvas(220);axes=[]
    gs=fig.add_gridspec(2,2,left=.18,right=.97,bottom=.11,top=.94,wspace=.59,hspace=.37)
    span=max(abs(d.normalized_difference_full).max(),abs(d.normalized_difference_local).max())*1.04
    for i,(proc,k) in enumerate([('calibrated',5),('global_train_only',5),('calibrated',10),('global_train_only',10)]):
        ax=fig.add_subplot(gs[i//2,i%2]);axes.append(ax);label(ax,chr(97+i))
        head(ax,('Participant calibration' if proc=='calibrated' else 'Training-global')+f' · top {k}')
        labels=[]
        for j,(loss,budget) in enumerate([(loss,b) for loss in EEGLOSS for b in [2,4,6]]):
            q=d[(d.processing==proc)&(d.k==k)&(d.contract==loss)&(d.budget==budget)].sort_values('subject');assert len(q)==106
            labels.append(short_loss(loss)+f' · {budget}')
            for ref,offy,side,mark in [('full',.15,1,'o'),('local',-.15,-1,'s')]:
                v=q['normalized_difference_'+ref]
                cloud(ax,v,11-j+offy,COL[ref],height=.25,side=side,tag=f'S12|{proc}|{k}|{loss}|{budget}|{ref}',
                      point_size=1.4,marker=mark,jitter=.04)
                ax.scatter(v.mean(),11-j+offy,s=8,marker='D' if ref=='full' else 's',
                     facecolors=COL[ref] if ref=='full' else 'white',edgecolors=COL[ref],lw=1,zorder=5)
        zero(ax);ax.set(xlim=(-span,span),ylim=(-.6,11.6),xlabel='Normalized loss difference')
        ax.set_yticks(range(11,-1,-1),labels);ax.xaxis.set_major_locator(MaxNLocator(4))
    fig.legend(handles=[Line2D([],[],marker='o',color=COL['full'],ls='',label='Versus complete'),
                        Line2D([],[],marker='s',color=COL['local'],mfc='white',ls='',label='Versus pairwise')],
        loc='lower center',bbox_to_anchor=(.58,.018),ncol=2,columnspacing=2)
    save(fig,'S12',axes,rows=[['a','b'],['c','d']],cols=[['a','c'],['b','d']])

def S13():uncertainty_family('S13',True)

def S14():
    d=data('S10_all_7320_frozen_label_subset_comparisons.csv');fig=canvas(199);axes=[]
    gs=fig.add_gridspec(3,2,left=.17,right=.97,bottom=.13,top=.93,wspace=.66,hspace=.42)
    lo=min(d.normalized_difference_full.min(),d.normalized_difference_local.min());hi=max(d.normalized_difference_full.max(),d.normalized_difference_local.max());pad=(hi-lo)*.06
    for i,budget in enumerate([2,4,6]):
        for j,proc in enumerate(['calibrated','global_train_only']):
            ax=fig.add_subplot(gs[i,j]);axes.append(ax);label(ax,chr(97+2*i+j))
            q=d[(d.processing==proc)&(d.budget==budget)];assert len(q)==1220
            labels=[]
            for k,n in enumerate(['5','10','All (21–22)']):
                v=q[q.label_availability==n].sort_values(['contract','fold','replicate']) if 'replicate' in q else q[q.label_availability==n]
                y=2-k;labels.append(('All 21–22' if k==2 else n+' labels')+f' · n={len(v)}')
                jitter=point_jitter(len(v),.045)
                for index,row in enumerate(v.itertuples()):
                    x0=row.normalized_difference_full;x1=row.normalized_difference_local;dy=jitter[index]
                    vertices=[(x0,y+.12+dy),(x0,y+.02+dy),(x1,y-.02+dy),(x1,y-.12+dy)]
                    ax.add_patch(PathPatch(MplPath(vertices,[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4]),
                        facecolor='none',edgecolor=P['mid'],lw=1,alpha=.035,zorder=1))
                for ref,dy,side,mark in [('full',.22,1,'o'),('local',-.22,-1,'s')]:
                    cloud(ax,v['normalized_difference_'+ref],y+dy,COL[ref],height=.27,side=side,
                          tag=f'S14|{proc}|{budget}|{n}|{ref}',point_size=1.7 if k<2 else 5,marker=mark,jitter=.045)
            zero(ax);ax.set(xlim=(lo-pad,hi+pad),ylim=(-.57,2.57))
            ax.set_yticks([2,1,0],labels);ax.xaxis.set_major_locator(MaxNLocator(4))
            if i==2:ax.set_xlabel('Normalized loss difference')
            head(ax,('Participant calibration' if proc=='calibrated' else 'Training-global')+f' · {budget} swaps')
    fig.legend(handles=[Line2D([],[],marker='o',color=COL['full'],ls='',ms=3.5,label='Versus complete top 10'),
                        Line2D([],[],marker='s',color=COL['local'],ls='',ms=3.5,label='Versus pairwise top 10')],
        loc='lower center',bbox_to_anchor=(.55,.028),ncol=2,columnspacing=1.8)
    save(fig,'S14',axes,rows=[['a','b'],['c','d'],['e','f']],cols=[['a','c','e'],['b','d','f']])

FUNCTIONS={k:globals()[k] for k in [f'M{i}' for i in range(1,6)]+[f'S{i}' for i in range(1,15)]}

def update_source_index():
    import hashlib
    rows=[]
    for p in sorted(SD.rglob('*.csv')):
        rows.append({'file':str(p.relative_to(SD)),'rows':len(pd.read_csv(p)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    atomic_bytes(SD/'SOURCE_DATA_INDEX.json',json.dumps(rows,indent=2).encode())

def overview():
    import fitz
    doc=fitz.open()
    for i in range(1,6):
        with fitz.open(ROOT/'MAIN_FIGURES'/f'Figure_{i}.pdf') as one:doc.insert_pdf(one)
    doc.set_metadata({'title':'Main figures 1–5','author':'','subject':'Local and complete information in measurement decisions',
                      'creator':'','producer':''})
    doc.set_toc([[1,f'Figure {i}',i] for i in range(1,6)])
    atomic_bytes(ROOT/'Main_Figures_Overview.pdf',doc.tobytes(garbage=4,deflate=True));doc.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',nargs='*',choices=list(FUNCTIONS),default=list(FUNCTIONS))
    parser.add_argument('--qa-dir',type=Path,default=ROOT.parent/'figure_quality_checks')
    args=parser.parse_args();E.ARGS=args
    for sub in ['MAIN_FIGURES','SUPPLEMENTARY_FIGURES','VECTOR_SOURCE','RASTER_SUBMISSION']:(ROOT/sub).mkdir(exist_ok=True)
    for key in args.only:
        E.configure_figure(key)
        FUNCTIONS[key]()
    update_source_index()
    if all((ROOT/'MAIN_FIGURES'/f'Figure_{i}.pdf').is_file() for i in range(1,6)):overview()
