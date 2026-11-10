"""Current five main figures: exact structures and existing empirical records."""
import figure_engine as E
from figure_engine import *
from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable

def off(ax):ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def source_label(s):return {'MEX':'MEx','UCI_HAR':'UCI HAR'}.get(s,s)
def arrow(ax,p,q,color=None,lw=1,rad=0):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='->',mutation_scale=7,lw=lw,
                 connectionstyle=f'arc3,rad={rad}',color=color or P['dark']))
def box(ax,x,y,w,h,text,face=None,fontsize=6.5):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.008,rounding_size=.012',
        facecolor=face or 'white',edgecolor=P['dark'],lw=1,zorder=3))
    ax.text(x,y,text,ha='center',va='center',fontsize=fontsize,zorder=4)

def M1():
    fig=canvas(178);a=fig.add_axes([.07,.56,.40,.37]);b=fig.add_axes([.58,.56,.36,.37]);c=fig.add_axes([.13,.11,.82,.30])
    for ax,ch in zip([a,b,c],'abc'):label(ax,ch)
    off(a);off(b);head(a,'A separate transformation for each state pair');head(b,'One transformation for every state')
    v=np.array([[.50,.86],[.17,.34],[.83,.34]])
    membership=data('M1_state_restriction_membership.csv')
    restrictions=membership[membership.kernel_scope.eq('specific_to_state_pair')].restriction.drop_duplicates()
    assert membership[membership.kernel_scope.eq('common_all_states')&membership.included].state.tolist()==[1,2,3]
    for restriction,col in zip(restrictions,[P['soft'],P['pink'],P['mid']]):
        i,j=(membership[(membership.restriction==restriction)&membership.included].state.to_numpy()-1)
        p,q=v[i],v[j];center=(p+q)/2;theta=np.degrees(np.arctan2(*(q-p)[::-1]))
        a.add_patch(Ellipse(center,np.linalg.norm(p-q)+.17,.22,angle=theta,
                    facecolor=col,edgecolor=P['dark'],alpha=.63,lw=1,zorder=1))
    for j,(x,y) in enumerate(v):
        a.scatter(x,y,s=360,marker=['o','s','^'][j],facecolor='white',edgecolor=P['dark'],lw=1,zorder=4)
        a.text(x,y,str(j+1),ha='center',va='center',fontsize=6.5,zorder=5)
    for x,y,t in [(.11,.69,'K₁₂, L₁₂'),(.90,.69,'K₁₃, L₁₃'),(.5,.14,'K₂₃, L₂₃')]:a.text(x,y,t,ha='center',fontsize=7)
    a.text(.50,.015,'Pairwise r₂ · maps may differ between pairs',ha='center',fontsize=6.5)
    for top,letter,direction in [(True,'K',1),(False,'L',-1)]:
        yy=[.86,.75,.64] if top else [.42,.31,.20]
        mid=np.mean(yy)
        b.add_patch(FancyBboxPatch((.445,min(yy)-.055),.11,max(yy)-min(yy)+.11,
                    boxstyle='round,pad=.008,rounding_size=.018',facecolor=P['soft'] if top else P['rose'],edgecolor=P['dark'],lw=1,zorder=3))
        b.text(.50,mid,letter,ha='center',va='center',fontsize=7,zorder=4)
        for k,y in enumerate(yy):
            left='P'+['₁','₂','₃'][k];right='Q'+['₁','₂','₃'][k]
            b.text(.15,y,left,ha='center',va='center',fontsize=7)
            b.text(.85,y,right,ha='center',va='center',fontsize=7)
            if direction>0:arrow(b,(.23,y),(.43,y));arrow(b,(.57,y),(.77,y))
            else:arrow(b,(.43,y),(.23,y),P['edge']);arrow(b,(.77,y),(.57,y),P['edge'])
    b.text(.5,.015,'Complete Δ · the same K and L across rows',ha='center',fontsize=6.5)
    fig.text(.52,.477,'3 outcomes · any finite state set:  r₂ = 0  ⇒  Δ = 0',ha='center',fontsize=7)
    d=data('M1_exact_curves.csv');assert (d.select_dtypes('number')>0).all().all()
    t=d.t.to_numpy();lo=np.log10(t.min());span=np.log10(t.max())-lo
    E.smooth_coordinate_ribbon(c,t,d.local_pairwise_deficiency,d.full_deficiency,
         (np.log10(t)-lo)/span,E.PARAM_CMAP,Normalize(0,1),alpha=.65)
    c.plot(t,d.full_deficiency,color=P['dark'],lw=1.8,label='Complete Δ · linear in t')
    c.plot(t,d.local_pairwise_deficiency,color=P['mid'],lw=1.8,ls='--',label='Pairwise r₂ · quadratic in t')
    c.set(xscale='log',yscale='log',xlim=(8e-5,.13),ylim=(1e-9,.1),xlabel='Deformation t',ylabel='Bidirectional deficiency')
    formatter=FuncFormatter(lambda v,pos:'10'+str(int(round(np.log10(v)))).translate(str.maketrans('0123456789-','⁰¹²³⁴⁵⁶⁷⁸⁹⁻')))
    c.xaxis.set_major_formatter(formatter);c.yaxis.set_major_formatter(formatter);c.minorticks_off()
    c.legend(loc='upper left',ncol=2,columnspacing=2)
    c.text(.98,.06,'Δ / √r₂ → 2/3',transform=c.transAxes,ha='right',fontsize=7)
    head(c,'Exact zero completeness does not give a linear near-zero rate')
    save(fig,'M1',[a,b,c],rows=[['a','b']],cols=[])

def polygon_probability_shading(ax,xy,cmap,norm):
    """Exact affine hull; shade by its ordinate p3, never by a fitted response."""
    xy=np.asarray(xy,float);yl,yh=xy[:,1].min(),xy[:,1].max()
    if yh-yl<1e-12:return
    def xs_at(y):
        out=[]
        for p,q in zip(xy,np.roll(xy,-1,axis=0)):
            if abs(q[1]-p[1])<1e-14:
                if abs(y-p[1])<1e-10:out.extend([p[0],q[0]])
            elif min(p[1],q[1])-1e-12<=y<=max(p[1],q[1])+1e-12:
                out.append(p[0]+(q[0]-p[0])*(y-p[1])/(q[1]-p[1]))
        return min(out),max(out)
    yy=np.linspace(yl,yh,65);xx=np.array([xs_at(y) for y in yy])
    vertices_x=xx.ravel();vertices_y=np.repeat(yy,2);p3=vertices_y*2/np.sqrt(3)
    tri=[]
    for i in range(len(yy)-1):tri.extend([[2*i,2*i+1,2*i+2],[2*i+1,2*i+3,2*i+2]])
    artist=ax.tripcolor(E.Triangulation(vertices_x,vertices_y,np.asarray(tri)),p3,shading='gouraud',cmap=cmap,norm=norm,alpha=.68,zorder=1)
    artist.set_clip_path(Polygon(xy,closed=True,transform=ax.transData))
    E.register_coordinate_gradient(artist,'y',yy,yy*2/np.sqrt(3),cmap,norm)

def channel_geometry(ax,d,opt,shade=True):
    z=d[d.candidate.eq(opt)]
    for exp,cmap,ls,filled in [('P',E.BLUE_CMAP,'--',False),('Q',E.ROSE_CMAP,'-',True)]:
        rows=z[z.experiment.eq(exp)].sort_values('state');xy=rows[['barycentric_x','barycentric_y']].to_numpy()
        if shade:polygon_probability_shading(ax,xy,cmap,Normalize(.26,.40))
        if opt=='A':ax.add_patch(Polygon(xy,closed=True,fill=False,edgecolor=P['dark'] if exp=='P' else P['rose_dark'],ls=ls,lw=1.4,zorder=3))
        elif exp=='Q':ax.plot(xy[:,0],xy[:,1],color=P['rose_dark'],lw=1.4,zorder=3)
        for k,r in enumerate(rows.itertuples()):
            ax.scatter(r.barycentric_x,r.barycentric_y,s=39 if filled else 48,marker=['o','s','^'][k],
                facecolor=cmap(Normalize(.26,.40)(r.p3)) if filled else 'white',edgecolor=P['dark'],lw=1,zorder=6 if filled else 5)
    for state,g in z.groupby('state'):
        assert len(g)==2
        ax.plot(g.barycentric_x,g.barycentric_y,color=P['grey'],lw=1,ls=':',alpha=.72,zorder=2)

def M2():
    d=data('M2_exact_channel_rows.csv');fig=canvas(172)
    a=fig.add_axes([.08,.38,.48,.50]);z=fig.add_axes([.14,.115,.33,.17]);b=fig.add_axes([.73,.60,.24,.28]);c=fig.add_axes([.73,.17,.24,.27])
    for ax,ch in [(a,'a'),(b,'b'),(c,'c')]:label(ax,ch)
    channel_geometry(a,d,'A');a.set(xlim=(.425,.575),ylim=(.215,.365),xlabel='p₂ + p₃/2',ylabel='√3 p₃/2')
    a.set_xticks([.45,.50,.55]);a.set_yticks([.24,.29,.34]);head(a,'A · exact family at t = 0.1')
    channel_geometry(z,d,'B',shade=False);z.set(xlim=(.483,.517),ylim=(.278,.300),xlabel='p₂ + p₃/2')
    z.set_xticks([.49,.50,.51]);z.set_yticks([.28,.29,.30]);head(z,'B · magnified, same probability coordinates')
    z.text(.5,.12,'P: three coincident state rows',transform=z.transAxes,ha='center',fontsize=6.5)
    for xx,cmap,name in [(.588,E.BLUE_CMAP,'P'),(.622,E.ROSE_CMAP,'Q')]:
        ca=fig.add_axes([xx,.46,.011,.22]);cb=fig.colorbar(ScalarMappable(norm=Normalize(.26,.40),cmap=cmap),cax=ca)
        cb.set_ticks([.27,.33,.39] if name=='Q' else []);ca.set_title(name,fontsize=6.5,pad=5)
        if name=='Q':cb.set_label('p₃',labelpad=1,fontsize=6.5)
    scores=data('M2_exact_candidates.csv').set_index('candidate')
    for opt,co in [('A',P['rose_dark']),('B',P['dark'])]:
        row=scores.loc[opt];x=float(row.r2_local);y=float(row.delta_full)
        b.plot([0,x],[y,y],ls=':',color=co,lw=1);b.plot([x,x],[0,y],ls=':',color=co,lw=1)
        b.scatter(x,y,s=48,facecolor=co,edgecolor='white',lw=1,zorder=5)
        b.annotate(opt,(x,y),xytext=(6,3) if opt=='A' else (-12,8),textcoords='offset points',fontsize=7)
    b.plot([0,.012],[0,.012],ls='--',color=P['grey'],lw=1)
    b.set(xlim=(-.0005,.0122),ylim=(-.001,.041),xlabel='Pairwise r₂',ylabel='Complete Δ')
    b.set_xticks([0,1/430,1/100],['0','1/430','1/100']);b.set_yticks([0,.01,3/86],['0','1/100','3/86'])
    head(b,'One slot · opposite choices')
    values=[float(scores.loc[x,'bayes_signed_risk_change']) for x in ['A','B']]
    norm=TwoSlopeNorm(vmin=min(values),vcenter=0,vmax=max(values))
    for i,val in enumerate(values):
        yy=np.linspace(min(0,val),max(0,val),49)
        xx=np.tile([i-.22,i+.22],len(yy));yt=np.repeat(yy,2);tri=[]
        for k in range(len(yy)-1):tri.extend([[2*k,2*k+1,2*k+2],[2*k+1,2*k+3,2*k+2]])
        artist=c.tripcolor(E.Triangulation(xx,yt,np.asarray(tri)),yt,shading='gouraud',cmap=E.SIGNED_CMAP,norm=norm,zorder=1)
        artist.set_clip_path(Rectangle((i-.22,min(0,val)),.44,abs(val),transform=c.transData))
        E.register_coordinate_gradient(artist,'y',yy,yy,E.SIGNED_CMAP,norm)
        c.scatter(i,val,s=25,color=P['rose_dark'] if val>0 else P['dark'],edgecolor=P['dark'],lw=1,zorder=5)
        c.annotate('+3/86' if i==0 else '−1/150',(i,val),xytext=(0,7 if i==0 else -12),textcoords='offset points',ha='center',fontsize=7)
    c.axhline(0,color=P['grey'],lw=1,ls='--');c.set(xlim=(-.6,1.6),ylim=(-.014,.044),ylabel='Bayes-risk change')
    c.set_xticks([0,1],['A\nPairwise choice','B\nComplete choice']);c.set_yticks([-.01,0,.02,.04])
    head(c,'Exact classification consequence')
    fig.legend(handles=[Line2D([],[],ls='--',marker='o',mfc='white',mec=P['dark'],color=P['dark'],label='Original P'),
       Line2D([],[],ls='-',marker='o',mfc=P['pink'],mec=P['dark'],color=P['rose_dark'],label='Replacement Q'),
       *[Line2D([],[],ls='',marker=m,mfc='white',mec=P['dark'],color=P['dark'],label=f'State {i+1}') for i,m in enumerate(['o','s','^'])]],
       loc='lower center',bbox_to_anchor=(.51,.017),ncol=5,columnspacing=1.1)
    save(fig,'M2',[a,b,c],rows=[],cols=[['b','c']],auxiliary_axes=[('a.B_magnification',z)])

STRATA=[('DSADS','repeated_subject_split','task','DSADS'),('WISDM','repeated_subject_split','task','WISDM'),
 ('MHEALTH','repeated_subject_split','task','MHEALTH'),('MEX','repeated_subject_split','task','MEx'),
 ('UCI_HAR','repeated_subject_split','task','UCI HAR · repeated'),('UCI_HAR','official_holdout','task','UCI HAR · official'),
 ('GasDrift','temporal_forward','task','GasDrift · tasks'),('GasDrift','temporal_forward','physical','GasDrift · slots')]
JOINT={('Lower','Lower'):P['dark'],('Lower','Higher'):P['rose_mid'],('Higher','Lower'):P['soft'],('Higher','Higher'):P['rose_dark']}
def paired_direction_colors(x,y):
    x=np.asarray(x);y=np.asarray(y);out=[]
    for a,b in zip(x,y):
        if abs(a)<=1e-12 or abs(b)<=1e-12:out.append(P['grey'])
        else:out.append(JOINT[('Lower' if a<0 else 'Higher','Lower' if b<0 else 'Higher')])
    return out

def M3():
    d=data('M3_paired_relative_and_retention.csv');g=data('M3_50pct_strata.csv');fig=canvas(178)
    a=fig.add_axes([.23,.55,.49,.36]);b=fig.add_axes([.12,.15,.33,.25]);c=fig.add_axes([.60,.15,.36,.25])
    for ax,ch in zip([a,b,c],'abc'):label(ax,ch)
    head(a,'Fixed 50% quota · six collections, eight strata');names=[]
    for j,(src,prot,unit,name) in enumerate(STRATA):
        q=d[(d.source==src)&(d.protocol==prot)&(d.unit==unit)].sort_values(['split','contract']);y=7-j
        r=g[(g.source==src)&(g.protocol==prot)&(g.unit==unit)].iloc[0]
        cloud(a,q.normalized_loss_difference,y,P['mid'],height=.66,point_size=5,jitter=.08,tag='M3|'+src+'|'+prot+'|'+unit)
        a.scatter(q.normalized_loss_difference.mean(),y-.02,s=23,marker='D',color=P['dark'],edgecolor='white',lw=1,zorder=5)
        for xx,col in [(1.17,'better_than_complete'),(1.42,'better_than_keep')]:a.text(xx,y,f'{int(r[col])}/{int(r.n)}',transform=a.get_yaxis_transform(),ha='center',va='center',fontsize=6.5)
        names.append(name+f' · n={len(q)}')
    for xx,t in [(1.17,'Lower than\ncomplete'),(1.42,'Lower than\nkeep')]:a.text(xx,1.025,t,transform=a.transAxes,ha='center',va='bottom',fontsize=6.5)
    lo,hi=d.normalized_loss_difference.min(),d.normalized_loss_difference.max();pad=(hi-lo)*.055
    zero(a);a.set(xlim=(lo-pad,hi+pad),ylim=(-.55,7.78),xlabel='Matched − complete loss / contract maximum')
    a.set_yticks(range(7,-1,-1),names);a.tick_params(axis='y',length=0)
    sx=max(abs(d.normalized_loss_difference))*1.08;sy=max(abs(d.normalized_keep_difference))*1.08
    b.add_patch(Rectangle((-sx,-sy),sx,sy,facecolor=P['pale'],edgecolor='none',lw=0,zorder=0))
    markers=['o','s','^','D','v','P']
    for src,marker in zip(SOURCE,markers):
        q=d[d.source.eq(src)]
        b.scatter(q.normalized_loss_difference,q.normalized_keep_difference,s=12,marker=marker,
                  c=paired_direction_colors(q.normalized_loss_difference,q.normalized_keep_difference),edgecolor=P['dark'],lw=1,alpha=.82,zorder=3)
    zero(b);zero(b,'y');head(b,'Paired decision-loss geometry')
    b.set(xlim=(-sx,sx),ylim=(-sy,sy),xlabel='Difference versus complete',ylabel='Difference versus keep');b.xaxis.set_major_locator(MaxNLocator(4));b.yaxis.set_major_locator(MaxNLocator(4))
    off(c);head(c,'The same 308 comparisons');j=data('M3_joint_outcome_counts.csv')
    rec=[((r.versus_complete_direction,r.versus_keep_direction),r.n,r.versus_complete_direction+'|'+r.versus_keep_direction) for r in j.itertuples()]
    colors={a+'|'+b:co for (a,b),co in JOINT.items()}
    alluvial(c,['Complete','Keep'],[['Lower','Higher']]*2,rec,tag='M3_joint_directions',bounds=(0,.02,1,.76),colors=colors,font=6.3,gap=.10)
    c.text(.22,.90,'Versus complete',ha='center',fontsize=6.5);c.text(.77,.90,'Versus keep',ha='center',fontsize=6.5)
    c.text(.5,-.08,'Numerical ties: 0',ha='center',fontsize=6.3)
    fig.legend(handles=[Line2D([],[],marker=m,ls='',ms=3.5,color=P['dark'],mfc='white',label=source_label(s)) for s,m in zip(SOURCE,markers)],
               loc='lower center',bbox_to_anchor=(.54,.025),ncol=6,columnspacing=.9,handletextpad=.35)
    save(fig,'M3',[a,b,c],rows=[['b','c']],cols=[])

RULES=[('train','Training','o','--'),('validation','Previous batch','s','-'),('blend','Blended','D',':'),('both_negative','Both negative','^','-.')]
def sign_flow(ax,df,tag,font=6.3):
    off(ax);records=[];colors={}
    for key,name,marker,ls in RULES:
        r=df[df.method.eq(key)].iloc[0];rate=float(r.false_harm/r.deployed)
        colors[key+'|higher']=E.ROSE_CMAP(rate)
        colors[key+'|other']=E.BLUE_CMAP(1-rate)
        records.extend([((name,'Higher loss'),r.false_harm,key+'|higher'),((name,'No higher loss'),r.no_higher_future_loss,key+'|other')])
    alluvial(ax,['Prediction rule','Future loss'],[[x[1] for x in RULES],['Higher loss','No higher loss']],records,
             tag=tag,colors=colors,font=font,gap=.075)
def harmful_colorbar(fig,pos):
    x,y,w,h=pos
    for dx,cmap,name in [(0,E.BLUE_CMAP,'No higher loss (%)'),(.56,E.ROSE_CMAP,'Higher loss (%)')]:
        cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=fig.add_axes([x+dx*w,y,.44*w,h]),orientation='horizontal')
        cb.set_ticks([0,.5,1],labels=['0','50','100']);cb.set_label(name,fontsize=6.5,labelpad=2)

def M4():
    fig=canvas(190);a=fig.add_axes([.12,.72,.84,.23]);b=fig.add_axes([.12,.39,.35,.25]);c=fig.add_axes([.61,.39,.35,.25]);d=fig.add_axes([.12,.15,.35,.17]);e=fig.add_axes([.61,.15,.35,.17])
    for ax,ch in zip([a,b,c,d,e],'abcde'):label(ax,ch)
    head(a,'Eight sensor slots · additive held-out loss');src=data('M4_six_physical_policies.csv').set_index('policy')
    for key,name,co,m,xy in [('full_fixed','Complete quota · 24/48',P['dark'],'s',(4.2,.0013)),('target_fixed','Goal quota · 22/48',P['pink'],'o',(4.05,-.0042)),('target_optional','Goal ceiling · 23/48',P['rose_mid'],'^',(2.05,.0028)),('validation_optional','Previous batch · 31/48',P['rose_dark'],'D',(3.1,-.0115)),('blend_optional','Blended ceiling · 28/48',P['rose'],'P',(3.35,-.0071)),('agreement_optional','Agreement ceiling · 25/48',P['soft'],'v',(.6,-.0061))]:
        r=src.loc[key];a.scatter(r.mean_selected,r.mean_risk_change,s=36,marker=m,facecolor=co,edgecolor=P['dark'],lw=1,zorder=4)
        a.annotate(name,(r.mean_selected,r.mean_risk_change),xytext=xy,fontsize=6.3,va='center',arrowprops={'arrowstyle':'-','color':P['grey'],'lw':1,'shrinkA':3,'shrinkB':4})
    a.scatter(0,0,s=25,marker='s',facecolor='white',edgecolor=P['black'],lw=1);a.annotate('Keep',(0,0),xytext=(.12,.0018),fontsize=6.5)
    zero(a,'y');a.set(xlim=(-.2,5.6),ylim=(-.0135,.0045),xlabel='Mean number of physical slots replaced',ylabel='Mean loss change versus keep');a.set_yticks([-.01,-.005,0]);a.set_xticks(range(6))
    flows=data('M4_actual_sign_transfer_flows.csv');rates=data('M4_all_batch_sign_transfers.csv')
    for ax,level,title in [(b,'task','Predicted benefits · task decisions'),(c,'physical','Predicted benefits · physical slots')]:sign_flow(ax,flows[flows.level.eq(level)],'M4_'+level);head(ax,title)
    for ax,level,title in [(d,'task','Task decisions across future batches'),(e,'physical','Physical slots across future batches')]:
        for key,name,m,ls in RULES:
            q=rates[(rates.level==level)&(rates.method==key)].sort_values('test_batch');y=q.fraction_predicted_beneficial_but_harmful
            ax.plot(q.test_batch,100*y,ls=ls,color=P['grey'],lw=1.2,zorder=2)
            ax.scatter(q.test_batch,100*y,s=22,marker=m,c=E.ROSE_CMAP(y),edgecolor=P['dark'],lw=1,zorder=4)
        ax.set(xlim=(6.75,10.25),ylim=(-4,104),xlabel='Held-out batch');ax.set_xticks([7,8,9,10]);ax.set_yticks([0,50,100]);head(ax,title)
    d.set_ylabel('Predicted benefits becoming harmful (%)')
    fig.legend(handles=[Line2D([],[],marker=m,ls=ls,color=P['grey'],mfc=P['pale'],mec=P['dark'],ms=4,label=name) for key,name,m,ls in RULES],loc='lower center',bbox_to_anchor=(.54,.065),ncol=4,columnspacing=1.5)
    harmful_colorbar(fig,[.31,.052,.47,.010])
    save(fig,'M4',[a,b,c,d,e],rows=[['b','c'],['d','e']],cols=[['b','d'],['c','e']])

def M5():
    design=data('M5_example_four_swap_design.csv');primary=data('M5_all_106_paired_participants.csv');persons=data('M5_all_false_alarm_participant_pairs.csv');stats=data('M5_all_false_alarm_conditions.csv')
    fig=canvas(200);a=fig.add_axes([.07,.735,.89,.22]);b=fig.add_axes([.12,.37,.31,.29]);c=fig.add_axes([.64,.37,.32,.29]);d=fig.add_axes([.27,.085,.68,.18])
    for ax,ch in zip([a,b,c,d],'abcd'):label(ax,ch)
    off(a);head(a,'Physical electrode choices and equal-search participant-disjoint evaluation')
    for row in design.itertuples():
        col=row.pair%2;rr=row.pair//2;x=.09+col*.18;y=.83-rr*.21
        for bit,name,xx in [(0,row.original_electrode,x-.035),(1,row.alternative_electrode,x+.035)]:
            selected=bit==row.selected_alternative
            a.scatter(xx,y,s=25,marker='o',facecolor=P['rose_mid'] if selected else 'white',edgecolor=P['dark'],lw=1.5 if selected else 1,zorder=4)
            a.text(xx,y+.045,name,ha='center',fontsize=6.3,zorder=5)
        a.plot([x-.019,x+.019],[y,y],color=P['mid'],lw=1)
    a.text(.18,-.02,'Eight active electrodes · four swaps',ha='center',fontsize=6.5)
    a.text(.70,.94,'2 / 4 / 6 swaps: 28 / 70 / 28 eligible designs',ha='center',fontsize=6.5)
    a.text(.70,.82,'Theta + alpha + beta · 3-class shrinkage LDA',ha='center',fontsize=6.5)
    for y,name,m,co in [(.64,'Training risk','o',P['pink']),(.42,'Complete Δ','s',P['dark']),(.20,'Pairwise r₂','^',P['mid'])]:
        a.scatter(.44,y,s=13,marker=m,facecolor=co,edgecolor=P['dark'],lw=1)
        a.text(.46,y,name,va='center',fontsize=6.5);arrow(a,(.60,y),(.65,y))
        box(a,.73,y,.14,.13,'Top 5 / 10',P['pale']);arrow(a,(.81,y),(.85,y));box(a,.92,y,.115,.13,'Choose 1')
    a.text(.92,.735,'Same labels',ha='center',fontsize=6.5)
    a.text(.70,-.02,'106 test participants · five disjoint folds',ha='center',fontsize=6.5)
    sx=max(abs(primary.target_minus_full))*1.08;sy=max(abs(primary.target_minus_local))*1.08
    b.add_patch(Rectangle((-sx,-sy),sx,sy,facecolor=P['pale'],edgecolor='none',lw=0,zorder=0))
    for fold,marker in zip(sorted(primary.fold.unique()),['o','s','^','D','v']):
        q=primary[primary.fold.eq(fold)]
        b.scatter(q.target_minus_full,q.target_minus_local,s=16,marker=marker,c=paired_direction_colors(q.target_minus_full,q.target_minus_local),edgecolor=P['dark'],lw=1,zorder=4)
    zero(b);zero(b,'y');b.set(xlim=(-sx,sx),ylim=(-sy,sy),xlabel='Loss difference versus complete',ylabel='Loss difference versus pairwise')
    b.xaxis.set_major_locator(MaxNLocator(4));b.yaxis.set_major_locator(MaxNLocator(4));head(b,'Calibrated · top 5 · false alarm ×3')
    b.legend(handles=[Line2D([],[],marker=m,ls='',ms=3.5,mfc='white',mec=P['dark'],color=P['dark'],label=str(int(f)+1)) for f,m in zip(sorted(primary.fold.unique()),['o','s','^','D','v'])],title='Test fold',loc='upper left',ncol=3,handlelength=1,columnspacing=.8,handletextpad=.4,title_fontsize=6.5)
    settings=[('calibrated',5),('calibrated',10),('global_train_only',5),('global_train_only',10)]
    vals=np.r_[persons.target_minus_full,persons.target_minus_local];pad=np.ptp(vals)*.055
    for j,(proc,k) in enumerate(settings):
        q=persons[(persons.processing==proc)&(persons.k==k)]
        for ref,dy,side,m in [('full',.18,1,'o'),('local',-.18,-1,'s')]:
            v=q['target_minus_'+ref];assert len(v)==106
            cloud(c,v,3-j+dy,COL[ref],side=side,height=.25,point_size=1.7,jitter=.045,marker=m,tag=f'M5|false_alarm_T2_x3|{proc}|{k}|{ref}')
            r=stats[(stats.processing==proc)&(stats.k==k)&(stats.reference==ref)].iloc[0]
            interval(d,r.mean_target_minus_ref,3-j+dy,r.ci_low,r.ci_high,COL[ref],m,ref=='local',3.8)
    labels=[('Calibrated' if p=='calibrated' else 'Global')+f' · {k}' for p,k in settings]
    c.set(xlim=(vals.min()-pad,vals.max()+pad),ylim=(-.6,3.6),xlabel='Individual loss difference');c.set_yticks(range(3,-1,-1),labels);zero(c);c.xaxis.set_major_locator(MaxNLocator(3));head(c,'False alarm ×3 · participant distributions')
    d.set(xlim=(stats.ci_low.min()-.002,stats.ci_high.max()+.002),ylim=(-.6,3.6),xlabel='Mean loss difference (95% paired interval)');d.set_yticks(range(3,-1,-1),labels);zero(d);d.xaxis.set_major_locator(MaxNLocator(5));head(d,'False alarm ×3 · all eight formal contrasts')
    fig.legend(handles=[Line2D([],[],marker='o',color=COL['full'],mfc=COL['full'],ls='-',label='Versus complete'),Line2D([],[],marker='s',color=COL['local'],mfc='white',ls='-',label='Versus pairwise')],loc='lower center',bbox_to_anchor=(.62,.016),ncol=2,columnspacing=2)
    save(fig,'M5',[a,b,c,d],rows=[['b','c']],cols=[])
