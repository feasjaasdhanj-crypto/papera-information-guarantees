#!/usr/bin/env python3
"""Deterministic vector plotting primitives and sparse native displays.

Density shapes are reproducible empirical layouts; no inferential quantities
are estimated. All model, policy, risk and interval inputs are formal objects.
"""
from pathlib import Path
import argparse, io, json, os, shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle, Polygon, Circle, FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.collections import Collection
from matplotlib.artist import Artist
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize, to_rgba
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
from matplotlib.tri import Triangulation
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42, 'font.size':7,
 'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
 'legend.fontsize':6.5,'axes.titlesize':7,'axes.titleweight':'normal',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,
 'ytick.major.size':2.5,'legend.frameon':False,'lines.linewidth':1.0,
 'text.color':'#272727','axes.labelcolor':'#272727','axes.edgecolor':'#4D4D4D',
 'xtick.color':'#272727','ytick.color':'#272727','savefig.facecolor':'white',
 'svg.hashsalt':'information-experiments-figures-v1'})

ROOT=Path(__file__).resolve().parents[1]
SD=ROOT/'SOURCE_DATA'
P={'dark':'#484878','mid':'#7884B4','soft':'#B4C0E4',
   'pale':'#E4E4F0','rose':'#E4CCD8','pink':'#F0C0CC',
   'edge':'#84546A','grey':'#767676','light':'#D8D8D8','black':'#272727'}
P['rose_mid']='#D38BA3';P['rose_dark']='#A65C79'
BLUE_CMAP=LinearSegmentedColormap.from_list('blue_purple_family',[P['pale'],P['soft'],P['mid'],P['dark']])
ROSE_CMAP=LinearSegmentedColormap.from_list('rose_family',[P['pale'],P['rose'],P['pink'],P['rose_mid'],P['rose_dark']])
SIGNED_CMAP=LinearSegmentedColormap.from_list('signed_loss_families',[
 (0,P['dark']),(.22,P['mid']),(.40,P['soft']),(.50,P['pale']),
 (.60,P['rose']),(.72,P['pink']),(.87,P['rose_mid']),(1,P['rose_dark'])])
PARAM_CMAP=LinearSegmentedColormap.from_list('ordered_parameter_families',[
 P['dark'],P['mid'],P['soft'],P['pale'],P['rose'],P['pink'],P['rose_mid'],P['rose_dark']])
HARM_NORM=TwoSlopeNorm(vmin=0,vcenter=.5,vmax=1)
CLOUD_KIND='loss';VALUE_NORM=TwoSlopeNorm(vmin=-.2,vcenter=0,vmax=.2)
COLOR_SETTING={}
COL={'full':P['dark'],'local':P['mid'],'target':P['pink']}
LOSS4=['balanced','miss3','falsealarm3','reject035']
EEGLOSS=['balanced','miss_T2_x3','false_alarm_T2_x3','reject035']
LOSSNAME={'balanced':'Equal classification cost','miss3':'Missed state ×3',
 'falsealarm3':'False alarm ×3','reject035':'Abstention (0.35)',
 'miss_T2_x3':'Missed state ×3','false_alarm_T2_x3':'False alarm ×3'}
SOURCE=['DSADS','WISDM','MEX','MHEALTH','UCI_HAR','GasDrift']

def data(n): return pd.read_csv(SD/n)
def canvas(h=110):return plt.figure(figsize=(7.086614173228347,h/25.4))
def label(ax,c):
    off=transforms.ScaledTranslation(-13/72,6/72,ax.figure.dpi_scale_trans)
    ax.text(0,1,c,transform=ax.transAxes+off,ha='left',va='bottom',
            fontsize=8,fontweight='bold')
def head(ax,text):ax.text(0,1.035,text,transform=ax.transAxes,ha='left',va='bottom',fontsize=7)
def zero(ax,axis='x'):
    if axis=='x':ax.axvline(0,color=P['grey'],lw=.65,ls=(0,(3,3)),zorder=0)
    else:ax.axhline(0,color=P['grey'],lw=.65,ls=(0,(3,3)),zorder=0)
def interval(ax,x,y,lo,hi,color,marker='o',open_mark=False,size=3.7):
    stroke=P['dark'] if color in [P['pale'],P['soft'],P['rose'],P['pink']] else color
    ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=marker,ms=size,
      mfc='white' if open_mark else color,mec=stroke,
      mew=1,color=stroke,ecolor=stroke,elinewidth=1,capsize=2,capthick=1,zorder=4)
def point_jitter(n,width=.16):
    # Geometry only: deterministic low-discrepancy stacking, no random sampling.
    return (((np.arange(n)*.6180339887498949)%1)-.5)*2*width
def marklegend(items,**kw):
    return [Line2D([],[],marker=m,ls=ls,color=c,mfc=c,mec=P['dark'],ms=4,
                   lw=1,label=t) for t,c,m,ls in items]
def atomic_bytes(path,content):
    temp=path.with_name(path.name+'.tmp')
    temp.write_bytes(content)
    os.replace(temp,path)

DENSITY_ROWS=[]
FLOW_AUDIT={}

def configure_figure(ident):
    """One actual-value color scale per figure; no method/category ordering."""
    global CLOUD_KIND,VALUE_NORM,COLOR_SETTING
    specs={
      'M3':('M3_paired_relative_and_retention.csv',['normalized_loss_difference']),
      'M5':('M5_all_false_alarm_participant_pairs.csv',['target_minus_full','target_minus_local']),
      'S2':('S11_information_landscape_and_gaps.csv',['complete_minus_pairwise']),
      'S3':('S2_all_924_budget_contrasts.csv',['normalized_difference']),
      'S4':('S3_UCI_all_paired_participants.csv',['full_minus_local']),
      'S6':('S5_all_one_slot_temporal_choices.csv',['normalized_future_change']),
      'S9':('EEG_primary_all_participant_budget_averages.csv',['target_minus_full','target_minus_local']),
      'S12':('S13_all_budget_specific_106_person_pairs.csv',['normalized_difference_full','normalized_difference_local']),
      'S13':('EEG_secondary_all_paired_participant_differences.csv',['loss_difference']),
      'S14':('S10_all_7320_frozen_label_subset_comparisons.csv',['normalized_difference_full','normalized_difference_local'])}
    CLOUD_KIND='gap' if ident=='S2' else 'loss'
    COLOR_SETTING={'figure':ident,'kind':CLOUD_KIND,'meaning':'Shading follows actual displayed values; shape, position and direct labels decode categories.','value_axis':'y' if ident=='S3' else 'x'}
    if ident in specs:
        filename,cols=specs[ident];values=data(filename)[cols].to_numpy(float)
        low,high=float(values.min()),float(values.max())
        if CLOUD_KIND=='gap':VALUE_NORM=Normalize(vmin=0,vmax=max(high,1e-12))
        else:
            span=max(abs(low),abs(high),1e-12)
            VALUE_NORM=TwoSlopeNorm(vmin=-span,vcenter=0,vmax=span)
        COLOR_SETTING.update(source=filename,fields=cols,vmin=VALUE_NORM.vmin,vmax=VALUE_NORM.vmax,center=0)
        if CLOUD_KIND=='loss':COLOR_SETTING['color_saturation_power']=1/3

def effect_colors(values,points=False):
    position=VALUE_NORM(values)
    if CLOUD_KIND=='loss':position=.5+.5*np.cbrt(2*(position-.5))
    col=(BLUE_CMAP if CLOUD_KIND=='gap' else SIGNED_CMAP)(position)
    if points:col[...,:3]*=.76
    return col

def value_ridge(ax,x,profile,baseline,side=1,height=.34,zorder=1):
    """32 vector strips color the unchanged256-coordinate empirical outline."""
    x=np.asarray(x);top=baseline+side*height*np.asarray(profile)/max(profile)
    for start in range(0,len(x)-1,8):
        end=min(start+8,len(x)-1);sl=slice(start,end+1)
        ax.fill_between(x[sl],baseline,top[sl],facecolor=effect_colors((x[start]+x[end])/2),
                        edgecolor='none',linewidth=0,alpha=1,antialiased=False,zorder=zorder)
    ax.plot(x,top,color=P['dark'],lw=1,alpha=.8,zorder=zorder+.2)

def vertical_value_ridge(ax,values,profile,baseline,width=.20,zorder=1):
    values=np.asarray(values);side=baseline+width*np.asarray(profile)/max(profile)
    for start in range(0,len(values)-1,8):
        end=min(start+8,len(values)-1);sl=slice(start,end+1)
        ax.fill_betweenx(values[sl],baseline,side[sl],facecolor=effect_colors((values[start]+values[end])/2),
                         edgecolor='none',lw=0,alpha=1,antialiased=False,zorder=zorder)
    ax.plot(side,values,color=P['dark'],lw=1,alpha=.8,zorder=zorder+.2)

def smooth_coordinate_ribbon(ax,x,lower,upper,values,cmap,norm,alpha=1,zorder=1):
    """One editable vector Gouraud mesh; no sampling/response estimate."""
    x=np.asarray(x,float);lower=np.broadcast_to(lower,x.shape);upper=np.broadcast_to(upper,x.shape)
    xx=np.repeat(x,2);yy=np.column_stack([lower,upper]).ravel();vv=np.repeat(np.asarray(values,float),2)
    triangles=[]
    for i in range(len(x)-1):triangles.extend([[2*i,2*i+1,2*i+2],[2*i+1,2*i+3,2*i+2]])
    artist=ax.tripcolor(Triangulation(xx,yy,np.asarray(triangles)),vv,shading='gouraud',cmap=cmap,norm=norm,alpha=alpha,zorder=zorder)
    clip=Polygon(np.vstack([np.column_stack([x,lower]),np.column_stack([x[::-1],upper[::-1]])]),closed=True,transform=ax.transData)
    artist.set_clip_path(clip)
    register_coordinate_gradient(artist,'x',x,values,cmap,norm)
    return artist

def register_coordinate_gradient(artist,axis,coordinates,values,cmap,norm):
    """Carry the identical one-coordinate field into a native SVG gradient."""
    coordinates=np.asarray(coordinates,float);values=np.asarray(values,float)
    assert axis in ['x','y'] and len(coordinates)==len(values)
    assert np.all(np.diff(coordinates)>0)
    artist._coordinate_gradient=(axis,coordinates,values,cmap,norm)

def native_coordinate_svg(svg,fig,artists):
    """Avoid triangle-compositing seams in SVG editors; keep exact vector clips.

    PDF retains its Gouraud mesh. SVG uses the same vertex colors, scalar
    coordinate, opacity and polygon clip as a native linearGradient.
    """
    if not artists:return svg
    ns='http://www.w3.org/2000/svg';tag=lambda name:'{'+ns+'}'+name
    ET.register_namespace('',ns)
    ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    root=ET.fromstring(svg)
    height=fig.get_figheight()*72;width=fig.get_figwidth()*72
    for artist in artists:
        axis,coords,values,cmap,norm=artist._coordinate_gradient
        group=next(g for g in root.iter(tag('g')) if g.get('id')==artist.get_gid())
        clips={g.get('clip-path') for g in group.iter() if g.get('clip-path')}
        assert len(clips)==1,(artist.get_gid(),clips)
        clip=clips.pop();ax=artist.axes
        xy=np.column_stack([coords,np.zeros_like(coords)]) if axis=='x' else np.column_stack([np.zeros_like(coords),coords])
        transform=ax.get_xaxis_transform() if axis=='x' else ax.get_yaxis_transform()
        pts=transform.transform(xy)*72/fig.dpi;pts[:,1]=height-pts[:,1]
        distance=pts[:,0] if axis=='x' else pts[:,1]
        offsets=(distance-distance[0])/(distance[-1]-distance[0])
        group.clear();group.set('id',artist.get_gid())
        defs=ET.SubElement(group,tag('defs'));gid=artist.get_gid()+'-fill'
        grad=ET.SubElement(defs,tag('linearGradient'),{'id':gid,'gradientUnits':'userSpaceOnUse',
            'x1':str(pts[0,0]),'y1':str(pts[0,1]),'x2':str(pts[-1,0]),'y2':str(pts[-1,1]),'color-interpolation':'sRGB'})
        for offset,rgba in zip(offsets,cmap(norm(values))):
            color='#'+''.join(f'{int(round(255*c)):02x}' for c in rgba[:3])
            ET.SubElement(grad,tag('stop'),{'offset':f'{offset:.12g}','stop-color':color,'stop-opacity':f'{rgba[3]:.12g}'})
        ET.SubElement(group,tag('rect'),{'x':'0','y':'0','width':str(width),'height':str(height),
            'fill':'url(#'+gid+')','opacity':str(artist.get_alpha() if artist.get_alpha() is not None else 1),
            'clip-path':clip})
    return '<?xml version="1.0" encoding="utf-8"?>\n'+ET.tostring(root,encoding='unicode')

def enforce_stroke_floor(fig):
    """NC's current line-art minimum is one physical point for visible strokes."""
    for artist in fig.findobj():
        if isinstance(artist,Line2D):
            if artist.get_linewidth()>0:artist.set_linewidth(max(1.,artist.get_linewidth()))
            if artist.get_markeredgewidth()>0:artist.set_markeredgewidth(max(1.,artist.get_markeredgewidth()))
        elif isinstance(artist,Collection):
            widths=np.asarray(artist.get_linewidths(),float)
            if len(widths):artist.set_linewidths(np.where(widths>0,np.maximum(widths,1.),0))
        elif hasattr(artist,'get_linewidth') and hasattr(artist,'set_linewidth'):
            try:
                w=artist.get_linewidth()
                if isinstance(w,(int,float)) and w>0:artist.set_linewidth(max(1.,w))
            except (TypeError,ValueError):pass

def display_density(values,tag):
    """Reproducible empirical display smoothing, not a response or CI estimate."""
    v=np.asarray(values,dtype=float)
    assert v.ndim==1 and np.isfinite(v).all()
    if len(v)<8 or np.ptp(v)<1e-12:return None
    sd=np.std(v,ddof=1);iqr=np.subtract(*np.percentile(v,[75,25]))/1.34
    scale=min(sd,iqr) if sd>0 and iqr>0 else max(sd,iqr)
    h=max(.9*scale*len(v)**(-.2),np.ptp(v)*1e-3)
    x=np.linspace(v.min(),v.max(),256)
    density=np.exp(-.5*((x[:,None]-v[None,:])/h)**2).mean(axis=1)/(h*np.sqrt(2*np.pi))
    assert np.isfinite(density).all() and density.max()>0
    for xx,yy in zip(x,density):DENSITY_ROWS.append(dict(group=tag,n=len(v),bandwidth=h,value=xx,display_density=yy))
    return x,density

def write_density_layout(ident):
    folder=SD/'DENSITY_LAYOUTS';folder.mkdir(exist_ok=True)
    if DENSITY_ROWS:
        temp=folder/(ident+'.csv.tmp')
        pd.DataFrame(DENSITY_ROWS).to_csv(temp,index=False,float_format='%.17g')
        temp.replace(folder/(ident+'.csv'))

def cloud(ax,values,y,color,height=.34,side=1,tag='',points=True,point_size=3.,marker='o',jitter=.085,point_color=None):
    v=np.asarray(values,float)
    density=display_density(v,tag)
    if density is not None:
        xx,dd=density
        value_ridge(ax,xx,dd,y,side=side,height=height)
    if points:
        yy=y-side*(.10+point_jitter(len(v),jitter))
        ax.scatter(v,yy,s=point_size,marker=marker,color=point_color if point_color else effect_colors(v,points=True),edgecolors='none',alpha=.82,zorder=3)
    return density

def paired_paths(ax,xx,yy,color,alpha=.10,linewidth=1):
    """Categorical identity connectors; no values exist between the columns."""
    xx=np.asarray(xx,float);yy=np.asarray(yy,float)
    assert len(xx)==len(yy) and np.isfinite(yy).all()
    for j in range(len(xx)-1):
        dx=xx[j+1]-xx[j]
        vertices=[(xx[j],yy[j]),(xx[j]+dx*.38,yy[j]),
                  (xx[j+1]-dx*.38,yy[j+1]),(xx[j+1],yy[j+1])]
        ax.add_patch(PathPatch(MplPath(vertices,[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4]),
                              fill=False,edgecolor=color,lw=linewidth,alpha=alpha,zorder=1))

def alluvial(ax,stages,orders,records,*,tag,bounds=(0,0,1,1),colors=None,
             node_labels=True,node_width=.025,gap=.05,alpha=.78,font=6.3,frequency_opacity=False):
    """Count-preserving bands between categorical memberships of the same records.

    records is [(node_tuple, integer_count, color_key)]. Every stage has the
    same total; widths use the same count scale at every stage.
    """
    bx,by,bw,bh=bounds
    records=[(tuple(p),int(n),c) for p,n,c in records if int(n)>0]
    assert records and all(len(p)==len(stages) for p,n,c in records)
    total=sum(n for p,n,c in records);ng=max(map(len,orders));unit=(1-gap*(ng-1))/total
    assert unit>0
    xs=np.linspace(.22,.77,len(stages)) if node_labels else np.linspace(.05,.95,len(stages))
    nodes=[];ports=[]
    for j,order in enumerate(orders):
        counts={key:sum(n for p,n,c in records if p[j]==key) for key in order}
        stackheight=unit*total+gap*(len(order)-1);top=(1+stackheight)/2
        here={};these={}
        for key in order:
            hh=counts[key]*unit;y0=top-hh;here[key]=(y0,hh,counts[key]);top=y0-gap
            selected=sorted([(i,p,n,c) for i,(p,n,c) in enumerate(records) if p[j]==key],
                            key=lambda a:tuple(str(k) for k in a[1][:j]+a[1][j+1:])+(str(a[3]),))
            at=y0
            for i,p,n,c in selected:these[i]=(at,at+n*unit);at+=n*unit
            assert abs(at-(y0+hh))<1e-9
        nodes.append(here);ports.append(these)
    for i,(path,n,ck) in enumerate(records):
        col=(colors or {}).get(ck,P['mid'])
        for j in range(len(stages)-1):
            x0=xs[j]+node_width/2;x1=xs[j+1]-node_width/2
            lo0,hi0=ports[j][i];lo1,hi1=ports[j+1][i];dx=x1-x0
            vertices=[(x0,lo0),(x0+.42*dx,lo0),(x1-.42*dx,lo1),(x1,lo1),
                      (x1,hi1),(x1-.42*dx,hi1),(x0+.42*dx,hi0),(x0,hi0),(x0,lo0)]
            verts=[(bx+bw*x,by+bh*y) for x,y in vertices]
            codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
                   MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
            aa=.30+.65*np.sqrt(n/total) if frequency_opacity else alpha
            ax.add_patch(PathPatch(MplPath(verts,codes),facecolor=col,edgecolor='none',lw=0,alpha=aa,zorder=1))
    for j,here in enumerate(nodes):
        for key,(yy,hh,n) in here.items():
            if n:
                ax.add_patch(Rectangle((bx+bw*(xs[j]-node_width/2),by+bh*yy),bw*node_width,bh*hh,
                    facecolor='white',edgecolor=P['dark'],lw=1,zorder=3))
            if node_labels:
                labeltext=str(key)+'\n'+f'n = {n:,}'
                if j==0:xt=xs[j]-.025;ha='right'
                elif j==len(stages)-1:xt=xs[j]+.025;ha='left'
                else:xt=xs[j];ha='center'
                ax.text(bx+bw*xt,by+bh*(yy+hh/2),labeltext,ha=ha,va='center',fontsize=font,zorder=5)
    FLOW_AUDIT[tag]={'total':total,'stages':stages,
        'node_counts':[{str(k):v[2] for k,v in d.items()} for d in nodes],
        'paths':[{'nodes':list(p),'count':n,'color_key':c} for p,n,c in records]}
    return nodes
def save(fig,ident,axes,rows=None,cols=None,exclude=None,exemptions=None,auxiliary_axes=None):
    ident=ident.upper()
    enforce_stroke_floor(fig)
    gradients=[a for a in fig.findobj() if hasattr(a,'_coordinate_gradient')]
    for i,artist in enumerate(gradients):artist.set_gid(f'coordinate-gradient-{i+1}')
    qa=ARGS.qa_dir;qa.mkdir(parents=True,exist_ok=True)
    ids=[chr(97+i) for i in range(len(axes))]
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids,
        row_groups=rows,column_groups=cols,exclude_axes=exclude or [],
        exemptions=exemptions or [],json_out=qa/f'{ident}.alignment.json',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    traces=[]
    for ax,panel in list(zip(axes,ids))+[(ax,panel) for panel,ax in (auxiliary_axes or [])]:
        clouds=[]
        for collection in ax.collections:
            if collection.__class__.__name__!='PathCollection':continue
            offsets=np.asarray(collection.get_offsets(),dtype=float)
            if offsets.size:
                assert np.isfinite(offsets).all()
                if ax.axison:
                    lo,hi=sorted(ax.get_xlim());bot,top=sorted(ax.get_ylim())
                    assert offsets[:,0].min()>=lo-1e-10 and offsets[:,0].max()<=hi+1e-10,(ident,panel,'x clipping')
                    assert offsets[:,1].min()>=bot-1e-10 and offsets[:,1].max()<=top+1e-10,(ident,panel,'y clipping')
                clouds.append(offsets.tolist())
        traces.append({'panel':panel,'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),
            'point_clouds':clouds,'lines':[{'x':np.asarray(l.get_xdata(),dtype=float).tolist(),
            'y':np.asarray(l.get_ydata(),dtype=float).tolist()} for l in ax.lines],
            'text':[t.get_text() for t in ax.texts]})
    atomic_bytes(qa/f'{ident}.plotted_elements.json',json.dumps(traces).encode())
    dest=ROOT/('MAIN_FIGURES' if ident.startswith('M') else 'SUPPLEMENTARY_FIGURES')
    name=('Figure_'+ident[1:] if ident.startswith('M') else 'Supplementary_Figure_'+ident[1:])
    pdf=dest/f'{name}.pdf'
    buf=io.BytesIO()
    fig.savefig(buf,format='pdf',metadata={'Title':'','Author':'','Subject':'','Keywords':'',
                            'Creator':'','Producer':'Matplotlib','CreationDate':None,'ModDate':None})
    atomic_bytes(pdf,buf.getvalue())
    assert pdf.stat().st_size>1000
    svg=io.StringIO()
    fig.savefig(svg,format='svg',metadata={'Creator':'','Date':None,'Title':''})
    atomic_bytes(ROOT/'VECTOR_SOURCE'/f'{name}.svg',native_coordinate_svg(svg.getvalue(),fig,gradients).encode())
    # High-quality review output; authoritative line-art submission is the vector PDF.
    if gradients:
        # Agg can omit a clipped Gouraud field on log axes. Render the verified
        # primary PDF so coordinate gradients agree in every submitted format.
        import fitz
        with fitz.open(stream=buf.getvalue(),filetype='pdf') as rendered:
            png_bytes=rendered[0].get_pixmap(dpi=450,alpha=False).tobytes('png')
    else:
        png=io.BytesIO()
        fig.savefig(png,format='png',dpi=450,metadata={'Software':'Matplotlib'})
        png_bytes=png.getvalue()
    atomic_bytes(ROOT/'RASTER_SUBMISSION'/f'{name}.png',png_bytes)
    write_density_layout(ident)
    atomic_bytes(qa/f'{ident}.flow_geometry.json',json.dumps(FLOW_AUDIT).encode())
    atomic_bytes(qa/f'{ident}.color_encoding.json',json.dumps(COLOR_SETTING).encode())
    DENSITY_ROWS.clear(); FLOW_AUDIT.clear()
    plt.close(fig)
    print('Rendered',ident,flush=True)


def composition_matrix():
    d=data('S4_all_concentration_contrasts.csv');triads=sorted(d.gases.unique())
    cols=[(mode,ref,loss) for mode in ['raw','anchor','common_equal'] for ref in ['full','local'] for loss in LOSS4]
    arr=np.array([[d[(d.gases==tri)&(d['mode']==mode)&(d.reference==ref)&(d.loss==loss)].normalized_difference.item()
                   for mode,ref,loss in cols] for tri in triads])
    assert arr.shape==(20,24) and np.isfinite(arr).all()
    fig=canvas(152);a=fig.add_axes([.13,.21,.72,.65]);cb=fig.add_axes([.88,.21,.018,.65])
    cmap=SIGNED_CMAP
    span=max(abs(arr.min()),abs(arr.max()))
    im=a.pcolormesh(np.arange(25)-.5,np.arange(21)-.5,arr,cmap=cmap,
                    norm=TwoSlopeNorm(vmin=-span,vcenter=0,vmax=span),shading='flat',rasterized=False)
    cell_labels=[]
    for (row,col),value in np.ndenumerate(arr):
        txt='0' if abs(value)<=1e-12 else f'{100*value:+.0f}'.replace('-', '−')
        rgba=cmap(im.norm(value));lum=.299*rgba[0]+.587*rgba[1]+.114*rgba[2]
        artist=a.text(col,row,txt,ha='center',va='center',fontsize=5.4,
                      color='white' if lum<.48 else P['black'])
        cell_labels.append((row,col,artist))
    a.set_ylim(19.5,-.5);a.set_xlim(-.5,23.5);a.set_yticks(np.arange(20),triads)
    a.set_xticks(np.arange(24),['E','M','F','A']*6)
    a.tick_params(axis='both',length=0,pad=4)
    for k,t in enumerate(['Observed composition','Single shared concentration','Equal common concentrations']):
        a.text(k*8+3.5,1.075,t,transform=a.get_xaxis_transform(),ha='center',fontsize=6.5)
        for j,lab in enumerate(['Complete','Pairwise']):
            a.text(k*8+j*4+1.5,-.09,lab,transform=a.get_xaxis_transform(),ha='center',fontsize=6.5)
    a.set_ylabel('Three-gas state set');a.set_xlabel('Comparator and loss contract · cell values ×100',labelpad=24)
    colorbar=fig.colorbar(im,cax=cb);colorbar.set_label('Normalized loss difference',fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for row,col,artist in cell_labels:
        box=artist.get_window_extent(renderer)
        corners=a.transData.transform([[col-.5,row-.5],[col+.5,row+.5]])
        low=corners.min(axis=0);high=corners.max(axis=0)
        assert box.x0>low[0] and box.x1<high[0] and box.y0>low[1] and box.y1<high[1]
    save(fig,'S5',[a],exclude=[cb])

def electrode_scores():
    d=data('S8_fixed_training_electrode_scores.csv');f=data('S8_five_fold_electrode_scores.csv')
    pairs=data('M5_eight_physical_pairs.csv');names=(pairs.original_electrode+' → '+pairs.alternative_electrode).tolist()
    fig=canvas(123);a=fig.add_axes([.19,.22,.32,.63]);b=fig.add_axes([.65,.22,.32,.63]);
    for ax,c in [(a,'a'),(b,'b')]:label(ax,c);ax.set(xlim=(0,.075),ylim=(-.5,7.65),xlabel='Training-estimated deficiency')
    head(a,'Fixed training split · alpha and beta');head(b,'Five training folds · beta')
    for j in range(8):
        for band,off,m in [('alpha',.14,'o'),('beta',-.14,'s')]:
            r=d[(d['index']==j)&(d.band==band)].iloc[0];y=7-j+off
            a.plot([r.local_r2,r.full_delta],[y,y],color=P['soft'],lw=1)
            a.scatter(r.local_r2,y,s=20,marker=m,facecolor='white',edgecolor=P['mid'],lw=.7)
            a.scatter(r.full_delta,y,s=8,marker=m,color=P['dark'],lw=.5)
        for h in range(5):
            r=f[(f['index']==j)&(f.fold==h)].iloc[0];y=7-j+(h-2)*.12
            b.plot([r.local_r2,r.full_delta],[y,y],color=P['soft'],lw=.7)
            b.scatter(r.local_r2,y,s=15,marker='o',facecolor='white',edgecolor=P['mid'],lw=.6)
            b.scatter(r.full_delta,y,s=6,marker='o',color=P['dark'],lw=.3)
    a.set_yticks(np.arange(8)[::-1],names);b.set_yticks(np.arange(8)[::-1],names)
    handles=[Line2D([],[],marker='o',color=P['mid'],mfc='white',ls='',ms=4,label='Pairwise r₂'),
             Line2D([],[],marker='o',color=P['dark'],ls='',ms=4,label='Complete Δ'),
             Line2D([],[],marker='o',color=P['grey'],ls='',ms=4,label='Alpha (fixed split)'),
             Line2D([],[],marker='s',color=P['grey'],ls='',ms=4,label='Beta (fixed split)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.57,.045),ncol=4,columnspacing=1)
    save(fig,'S11',[a,b],rows=[['a','b']],cols=[])
