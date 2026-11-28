"""Restyle the code's SVG output (figures_svg/) in the style of the published figures (figures_restyled/).
Usage: python restyle_all.py [name-substring ...]   (no argument: all figures)"""
from svgrestyle import *
import sys, os
HERE=os.path.dirname(os.path.abspath(__file__))
D=os.path.join(HERE,'..','..','figures_svg','')      # input: SVGs written by run_all_analyses.py
OUT=os.path.join(HERE,'..','')                       # output: figures_restyled/
only=sys.argv[1:]
def sync_markers_to_png(f,png):
    # keep only significance markers that are still drawn in the (newer) PNG; take their size from the PNG
    from scipy import ndimage as ndi
    a=np.array(Image.open(png).convert('RGB')).astype(int); nonw=a.sum(2)<735; k=300/72; kept=dropped=resized=0
    tmpl={}
    for A in f.axes:
        for art in A['arts']:
            if art['kind']=='markers':
                for (x,y,p,st) in art['pts']: tmpl[round(p.get_extents().width,1)]=p
    sizes=sorted(tmpl); small,big=tmpl[sizes[0]],tmpl[sizes[-1]]
    lab,n=ndi.label(nonw[:int(a.shape[0]*0.25)]); objs=ndi.find_objects(lab)
    for A in f.axes:
        for art in A['arts']:
            if art['kind']!='markers': continue
            new=[]
            for (x,y,p,st) in art['pts']:
                px,py=int(round(x*k)),int(round(y*k))
                l=lab[py,px] if py<lab.shape[0] else 0
                if l==0: dropped+=1; continue
                sl=objs[l-1]; w=sl[0].stop-sl[0].start
                p2=big if w>=13 else small
                if p2 is not p: resized+=1
                new.append((x,y,p2,st)); kept+=1
            art['pts']=new
    print('   markers kept',kept,'dropped',dropped,'resized',resized)
def drop_series(f,pattern):
    # remove a plotted series (line, its markers, legend entry) identified by its legend label
    cols=set()
    for A in f.axes:
        for arts,lab in (A['legend'] or []):
            if arts and re.search(pattern,lab): cols.add(arts[0]['st'].get('stroke').lower())
    n=0
    for A in f.axes:
        keep=[]
        for a in A['arts']:
            if a['kind']=='line' and a['st'].get('stroke','').lower() in cols: n+=1; continue
            if a['kind']=='markers' and a['pts'] and (a['pts'][0][3].get('fill') or '').lower() in cols: n+=1; continue
            keep.append(a)
        A['arts']=keep
        if A['legend']: A['legend']=[(arts,lab) for arts,lab in A['legend'] if not re.search(pattern,lab)]
    print('   dropped artists',n,cols)
COMPACT={'fig4b_glm','ext5a_glm_animal','fig4ce_rsa_1','ext5b_rsa_animal_1','ext7_dlc_rsa_1','fig6b_policy_glm','ext7_dlc_policy_glm','fig6c_policy_corr','ext5cd_policy_animal_1','ext7_dlc_policy_corr'}
def go(name,png_sync=None,drop=None,pre=None,**kw):
    if only and not any(o in name for o in only): return
    f=Fig(D+name+'.svg')
    if drop: drop_series(f,drop)
    if png_sync: sync_markers_to_png(f,png_sync)
    if pre: pre(f)
    if name in COMPACT and 'scale' in kw: kw['scale']*=(0.84 if 'policy_corr' in name or 'policy_animal_1' in name else 0.72)   # closer to the printed panel size
    if name.startswith('fig'):
        post0=kw.get('post')
        def post_main(F,axs,H,fig,post0=post0):
            if post0: post0(F,axs,H,fig)
            for ax in axs:
                ax.title.set_fontweight('normal')
                for t in (ax.title,ax.xaxis.label,ax.yaxis.label): t.set_text(sentence(t.get_text()))
                lg=ax.get_legend()
                if lg:
                    for t in lg.get_texts(): t.set_text(sentence(t.get_text()))
        kw['post']=post_main
    render(f,OUT+name,fmt=('svg',),**kw); print('ok',name)
def sentence(s):
    # main-figure labels are printed in sentence case ('Choice time', '% variance explained')
    out=[]; first=True
    for w in re.split(r'(\s+)',s):
        if not w.strip(): out.append(w); continue
        core=w.strip('()%.,')
        if w in ('x','X'): w='\u00d7'
        elif w=='(Normalised)': w='(normalized)'
        elif not first and len(core)>1 and core[0].isupper() and core[1:].islower() and core not in ('Init','Ch','Out'): w=w.replace(core,core.lower())
        if core: first=False
        out.append(w)
    return ''.join(out)
def thin_x(keep):
    def post(F,axs,H,fig):
        for ax in axs:
            tl=[(p,t.get_text()) for p,t in zip(ax.get_xticks(),ax.get_xticklabels())]
            kp=[(p,t) for p,t in tl if keep(t)]
            if kp: ax.set_xticks([p for p,_ in kp],[t for _,t in kp],fontsize=7)
    return post
def _iv(t):
    try: return int(t.replace('\u2212','-'))
    except Exception: return None
def post_solid(F,axs,H,fig):
    for ax in axs:
        for l in ax.lines:
            if l.get_linestyle() not in ('-','None','') and l.get_marker() in (None,'None',''): l.set_linestyle('-'); l.set_color(GREY); l.set_linewidth(1.0)
EVEN=thin_x(lambda t: _iv(t) is not None and _iv(t)%2==0)
ODD=thin_x(lambda t: _iv(t) is not None and _iv(t)%2==1)
def post_pct(F,axs,H,fig):
    for ax in axs:
        tl=[(p,t.get_text()) for p,t in zip(ax.get_yticks(),ax.get_yticklabels())]
        ax.set_yticks([p for p,_ in tl],['%g'%round(float(t.replace('\u2212','-'))*100) for _,t in tl],fontsize=7)
def stamp(ax,sy=-0.2): ax.text(1.0,sy,'(s)',transform=ax.transAxes,ha='center',va='top',fontsize=7)
REG={'#8c3b49':PFC,'#212053':PFC,'#c2abba':CA1,'#808080':GREY}
GREEN='#99c772'; PINK='#ef9bc2'; BGREEN='#829c81'; BPINK='#d571aa'
# ---------- Fig 4B / Ext 5A  GLM
def post_glm(F,axs,H,fig):
    for i,ax in enumerate(axs):
        xt=ax.get_xticks()
        for xi,c,ls in ((2,'#c8c8c8','-'),(3,INK,(0,(3,1.5))),(4,'#e28aa5','-')): ax.axvline(xt[xi],color=c,lw=0.5,ls=ls,zorder=0)   # initiation, choice, outcome as printed
        stamp(ax)
        if i: ax.set_ylabel('')
    hs=[Line2D([],[],color=CA1,lw=1,label='CA1'),Line2D([],[],color=PFC,lw=1,label='PFC'),
        Line2D([],[],ls='',marker='o',ms=1.0,mfc=PFC,mec='none',label='p < .05 PFC/CA1'),Line2D([],[],ls='',marker='o',ms=2.0,mfc=PFC,mec='none',label='p < .01 PFC/CA1'),
        Line2D([],[],ls='',marker='o',ms=1.0,mfc=GREY,mec='none',label='p < .05 Δ PFC and CA1'),Line2D([],[],ls='',marker='o',ms=2.0,mfc=GREY,mec='none',label='p < .01 Δ PFC and CA1')]
    axs[-1].legend(handles=hs,loc='upper left',bbox_to_anchor=(1.03,0.75),handlelength=1.4,labelspacing=0.25,borderaxespad=0)
go('fig4b_glm',scale=0.38,cmap=REG,legend='none',post=post_glm,xrot=r'Init|Ch|Out',ms=2.6)
def post_glm_animal(F,axs,H,fig):
    post_glm(F,axs,H,fig)
    hs=[Line2D([],[],color=CA1,lw=1,label='CA1'),Line2D([],[],color=PFC,lw=1,label='PFC'),
        Line2D([],[],ls='',marker='o',ms=2.2,mfc=GREY,mec='none',label='p < .05 Δ PFC and CA1')]
    axs[-1].legend(handles=hs,loc='upper left',bbox_to_anchor=(1.03,0.75),handlelength=1.4,labelspacing=0.25,borderaxespad=0)
def only_region_diff(f,top=7.0):
    # as printed: Ext 5a carries only the CA1 vs PFC (animal permutation) markers, on one row; the within-region rows are left out
    for A in f.axes:
        keep=[]
        for a in A['arts']:
            if a['kind']=='markers':
                fill=(a['pts'][0][3].get('fill') or '').lower() if a['pts'] else ''
                print('   marker fill',fill,len(a['pts']))
                if fill not in ('#808080','grey','gray'): continue
                a['pts']=[(p[0],A['bbox'][2]+top)+tuple(p[2:]) for p in a['pts']]
            keep.append(a)
        A['arts']=keep
go('ext5a_glm_animal',pre=only_region_diff,scale=0.38,cmap=REG,legend='none',post=post_glm_animal,xrot=r'Init|Ch|Out',ms=6.5)
# ---------- Fig 4E / Ext 5B  RSA CPD time courses
def post_rsa1(F,axs,H,fig):
    for i,ax in enumerate(axs):
        xt=list(ax.get_xticks()); lab=[t.get_text() for t in ax.get_xticklabels()]
        if len(xt)==3 and lab[-1]=='+0.5':     # the code's panel stops labelling at +0.5; the printed one also marks +1
            ax.set_xticks(xt+[xt[1]+2*(xt[2]-xt[1])],lab+['+1'])
            for l in ax.get_xticklabels():
                if re.search(r'Ch|Out',l.get_text()): l.set_rotation(45); l.set_ha('right'); l.set_rotation_mode('anchor')
                else: l.set_rotation(0); l.set_ha('center'); l.set_fontsize(6)
        stamp(ax,-0.24)
        if i: ax.set_ylabel('')
RT=['Port','A vs B Choice','Outcome','Outcome\nat A vs B','Choice vs\nInitiation','Problem Specific\nA Choice']
for n,m in (('fig4ce_rsa_1',1.8),('ext5b_rsa_animal_1',3.6)):  # Ext 5b carries only the few animal-permutation markers: drawn larger
    go(n,scale=0.72,cmap=REG,ms=m*0.72,post=post_rsa1,xrot=r'Ch|Out',titles=RT,gap=(1.05,1))
# ---------- Fig 4C / Ext 5B heatmaps
LAB=[(r'^([AB]) T(\d) R$',r'\1 reward \2'),(r'^([AB]) T(\d) NR$',r'\1 non-rew. \2'),(r'^I? ?T(\d)$',r'Initiation \1'),(r'^(CA1|HP|PFC) Reward Time$',r'\1\nat Outcome Time'),(r'^(CA1|HP|PFC) Choice Time$',r'\1\nat Choice Time')]
for n in ('fig4ce_rsa_2','ext5b_rsa_animal_2'):
    go(n,scale=0.36,text=LAB+[(r'^HP\b','CA1')],gap=(1.25,1.35))
for n in ('fig4ce_rsa_3','ext5b_rsa_animal_3'):
    go(n,scale=0.42,noticks=True,titles=['Port','A vs B Choice','Outcome','Outcome\nat A vs B','Choice vs\nInitiation','Problem Specific\nA Choice'])
# ---------- Fig 4F,G decoding
def post_bar(F,axs,H,fig):
    # as printed: a coloured box around each pair of bars, square animal markers, and the CA1 vs PFC test above each pair
    # (animal permutation test, 35 splits: Correct 2/35, Position 1/35, Port 8/35)
    from matplotlib.patches import Rectangle
    ax=axs[0]; y0=ax.get_ylim()[0]                      # SVG coordinates: y grows downwards, y0 is the axis floor
    bars=sorted([p for p in ax.patches if isinstance(p,PathPatch)],key=lambda p:p.get_extents().x0)
    ext=[p.get_path().get_extents() for p in bars]; pts=[l for l in ax.lines if l.get_marker()=='o']
    for l in pts: l.set_marker('s'); l.set_markersize(l.get_markersize()*0.9); l.set_markerfacecolor('#9d9d9d')
    for g,(col,ls,lab) in enumerate((('#3bb5e5','-','NS'),('#d3202f','-','*'),('#8fb93a',(0,(3,2)),'NS'))):
        e=ext[2*g:2*g+2]; xa=min(b.x0 for b in e); xb=max(b.x1 for b in e); pad=(xb-xa)*0.04
        ys=[b.y0 for b in e]+[float(l.get_ydata()[0]) for l in pts if xa<=float(l.get_xdata()[0])<=xb]
        top=min(ys)-14
        ax.add_patch(Rectangle((xa-pad,top-24),xb-xa+2*pad,y0-(top-24),fc='none',ec=col,lw=1.0,ls=ls,zorder=6,clip_on=False))
        c=[(b.x0+b.x1)/2 for b in e]
        ax.plot([c[0],c[0],c[1],c[1]],[top+4,top,top,top+4],color='#808080',lw=0.6,zorder=7,clip_on=False)
        ax.text((c[0]+c[1])/2,top-(2 if lab=='NS' else -3),lab,ha='center',va='bottom',fontsize=7 if lab=='NS' else 9,color='#808080',zorder=7)
    axs[0].legend(handles=[Patch(fc=CA1,label='CA1'),Patch(fc=PFC,label='PFC')],loc='upper left',bbox_to_anchor=(1.04,1.0),handlelength=2.2,borderaxespad=0)
go('fig4fg_decode_2',scale=0.55,cmap={'#66c2a5':PFC,'#fc8d62':CA1,'#000000':GREY},solid=True,scatter_color='#595a5c',legend='none',post=post_bar,lw={'line':1.0,'lines':1.0},ylabel='Decoding (%)',xlabel='',text=[(r'^Area$',''),(r'^HP$','CA1'),(r'^Position in Trial Sequence$','Position in\ntrial\nsequence')])
# ---------- Fig 5 SVD
go('fig5ac_svd_examples_1',scale=0.6,cmap={'#008000':GREEN,'#ffc0cb':PINK},xrot=r'Init|Ch|Out',titles=[f'First Temporal Pattern\nin Problem {i}' for i in (1,2,3)],ylabel=['a.u.','',''],xlabel='Time in trial',gap=(1.1,1))
go('fig5ac_svd_examples_2',scale=0.6,titles=[f'First Cellular Pattern\nin Problem {i}' for i in (1,2,3)],ylabel=['a.u.','',''],gap=(1.1,1))
SV={'#008000':PFC,'#00ff00':'#8696b2','#000000':'#d8b3bb','#808080':'#cac5d5'}
def svt(sh): return [(r'^Within (PFC|HP)$',r'\1 same problem\ncross-validated'),(r'^Between (PFC|HP)$',r'\1 different problem'),(r'^perm (PFC|HP)$',r'\1 %s shuffle'%sh),
   (r'^Temporal Modes$','Temporal singular vectors'),(r'^Cellular Modes$','Cellular singular vectors'),(r'^Variance Explained$','% Variance Explained'),
   (r'^Cumulative Weight$','Cumulative Weight (Normalised)'),(r'^Cellular and Temporal Modes$','Cellular and temporal singular vectors'),(r'^Cellular and Temporal Modes with (\w+) Shuffle$',r'Cellular and temporal singular vectors\n(\1 shuffle)'),
   (r'^Number of (.*) \((.*)\)$',r'Number of \1\n(\2)')]
def post_ca1(F,axs,H,fig):
    for ax in axs:
        lg=ax.get_legend()
        if lg:
            hs=lg.legend_handles; labs=[t.get_text().replace('HP','CA1') for t in lg.get_texts()]
            key=lambda l: (2 if 'shuffle' in l else 1 if 'different' in l else 0, 0 if l.startswith('PFC') else 1)
            o=sorted(range(len(labs)),key=lambda i: key(labs[i])); lg.remove()
            ax.legend([hs[i] for i in o],[labs[i] for i in o],loc='best',handlelength=1.6,labelspacing=0.2,borderaxespad=0.6)   # inside the panel, as printed
def post_ca1_out(F,axs,H,fig):
    # Fig 5g with the shuffles: no free space inside the panel, legend stays beside it
    post_ca1(F,axs,H,fig)
    for ax in axs:
        lg=ax.get_legend()
        if lg: lg.set_loc('upper left'); lg.set_bbox_to_anchor((1.02,1.0))
for n,sh in (('fig5dh_svd_main_1','time'),('fig5dh_svd_main_2','cell'),('fig5dh_svd_main_3','time'),('fig5dh_svd_main_4','cell')):
    go(n,scale=0.6,cmap=SV,text=svt(sh),legend='right',post=post_ca1_out if n[-1] in '34' else post_ca1,lw={'line':1.2})
go('fig5de_svd_brains',scale=0.6,cmap={'#008000':'#0b918f','#ffc0cb':'#cac5d5'},text=[(r'^Temporal Modes$','Temporal singular vectors between\nregions')]+svt('time'),legend='best',lw={'line':1.2},legend_kw=dict(loc='lower right',borderaxespad=0.6))
HI={'#808080':GREY,'#ffc0cb':'#d8b1ad'}
HT=[(r'^Cellular vectors$','Cellular singular vectors'),(r'^Cellular and temporal singular vectors$','Cellular and temporal singular\nvectors'),
    (r'^Policy on (A|B) difference between CA1 and PFC$',r'Policy on \1 Diagonal Sum\nDifference between CA1\nand PFC')]
HS=0.72   # permutation-test histograms: drawn large enough to read the two lines
def post_hist(F,axs,H,fig):
    for i,ax in enumerate(axs):
        if i: ax.set_ylabel('')
    hs=[Line2D([],[],color=INK,lw=2,label='95% confidence interval'),Line2D([],[],color='#d8b1ad',lw=2,label='Difference in the data')]
    axs[-1].legend(handles=hs,loc='upper left',bbox_to_anchor=(1.03,1.0),handlelength=0.8,labelspacing=0.25,borderaxespad=0)
def post_hist_perm(F,axs,H,fig):
    post_hist(F,axs,H,fig)
    # when the data line sits exactly on the threshold line, draw it dashed on top so both stay visible
    import matplotlib.colors as mc
    for ax in axs:
        vl=[l for l in ax.lines if len(l.get_xdata())==2 and l.get_xdata()[0]==l.get_xdata()[1]]
        dat=[l for l in vl if mc.to_hex(l.get_color())=='#d8b1ad']; thr=[l for l in vl if mc.to_hex(l.get_color())!='#d8b1ad']
        for d in dat:
            if any(abs(d.get_xdata()[0]-t.get_xdata()[0])<1.5 for t in thr): d.set_zorder(10); d.set_linestyle((0,(2,2)))
    hs=[Line2D([],[],color=INK,lw=2,label='Permutation threshold (P < .05)'),Line2D([],[],color='#d8b1ad',lw=2,label='Difference in the data')]
    axs[-1].legend(handles=hs,loc='upper left',bbox_to_anchor=(1.03,1.0),handlelength=0.8,labelspacing=0.25,borderaxespad=0)
go('fig5dh_svd_main_5',scale=HS,cmap=HI,text=HT,legend='none',lw={'lines':2.6},post=post_hist_perm,gap=(1.2,1))
for n in [p+str(i) for p in ('ext9a_svd_all_','ext9b_svd_init_choice_','ext9c_svd_dlc_','ext9d_svd_animal_','ext9d_svd_animal_all_') for i in (1,2,3)]:
    go(n,scale=0.6,cmap=SV,text=svt('time'),legend='right',post=post_ca1,lw={'line':1.2})
for n in ('fig6d_policy_diag_2','ext9a_svd_all_4','ext9b_svd_init_choice_4','ext9c_svd_dlc_4','ext9d_svd_animal_4','ext9d_svd_animal_all_4'):
    go(n,scale=HS,cmap=HI,text=HT,legend='none',lw={'lines':2.6},post=post_hist_perm,gap=(1.2,1))
def post_hist_5c(F,axs,H,fig):
    # Ext 5c: with 35 animal splits the threshold is the second-largest split; say in words where the data falls
    post_hist_perm(F,axs,H,fig)
    for ax,t in zip(axs,('Data: 2nd largest\nof 35 splits\nP = 2/35 = .057\nnot significant','Data: largest\nof 35 splits\nP = 1/35 = .029')):
        ax.text(0.03,0.97,t,transform=ax.transAxes,ha='left',va='top',fontsize=7)
go('ext5cd_policy_animal_3',scale=HS,cmap=HI,text=HT,legend='none',lw={'lines':2.6},post=post_hist_5c,gap=(1.2,1))
# ---------- Fig 6 policy
go('fig6a_policy_rec',scale=0.5,cmap={'#b6aca6':BGREEN,'#212053':BPINK,'#1f77b4':GREY},lw={'lines':0.5},titles=['History of Choices','History of\nReward X Choice'],ylabel=['Regression Coefficient']*2,xlabel='N-back Trials')
POL={r'^Reward x Choice$':'#7db7cf',r'^Choice$':'#c2acbb',r'^Reward$':'#8c3c4a',r'^Policy$':'#252552',r'^Policy x Choice$':'#b9a658',r'^Prev Choice$':'#9d9d9d'}
def post_pol(F,axs,H,fig):
    for ax in axs: stamp(ax,-0.16)
def post_pol_swap(F,axs,H,fig):
    # the paper shows CA1 on the left and PFC on the right; the code draws PFC first
    post_pol(F,axs,H,fig)
    p0,p1=axs[0].get_position(),axs[1].get_position(); axs[0].set_position(p1); axs[1].set_position(p0)
    for ax in axs:
        lg=ax.get_legend()
        if lg is not None:
            hs=lg.legend_handles; labs=[t.get_text() for t in lg.get_texts()]; lg.remove()
            axs[0].legend(hs,labs,loc='upper left',bbox_to_anchor=(1.02,1.0),handlelength=1.4,labelspacing=0.2,borderaxespad=0); break
go('fig6b_policy_glm',drop=r'^Prev Choice$',scale=0.45,ms=2.4,by_label=POL,legend='right',xrot=r'Init|Ch|Out',post=post_pol,text=[(r'^Reward x Choice$','Reward x Choice'),(r'^Reward$','Reward')])
def post_corr(F,axs,H,fig):
    for i,ax in enumerate(axs):
        stamp(ax,-0.22)
        if i: ax.set_ylabel('')
    drawn=set()
    for ax in axs:
        for l in ax.lines:
            if l.get_marker()=='o': drawn.add(matplotlib.colors.to_hex(l.get_markerfacecolor()))
    hs=[h for h in H[-1][1] if h.get_marker() in (None,'None','') or matplotlib.colors.to_hex(h.get_markerfacecolor()) in drawn]
    # as printed: legend in one band above the panels, y axis in steps of 0.1
    F.legend(handles=hs,loc='lower center',bbox_to_anchor=(0.5,1.06),ncol=len(hs),handlelength=1.6,columnspacing=1.2,borderaxespad=0,fontsize=7)
    for ax in axs:
        yt=list(ax.get_yticks()); lab=[float(t.get_text().replace('\u2212','-')) for t in ax.get_yticklabels()]
        k=(yt[1]-yt[0])/(lab[1]-lab[0]); lo,hi=sorted(ax.get_ylim()); vals=[v/10 for v in range(-5,15)]
        keep=[(yt[0]+(v-lab[0])*k,v) for v in vals if lo<=yt[0]+(v-lab[0])*k<=hi]
        ax.set_yticks([p for p,_ in keep],['%g'%v for _,v in keep],fontsize=7)
def post_corr_inset(mat):
    # as printed: a small copy of a policy correlation matrix in each panel, with a red line at the time the slice is taken
    def post(F,axs,H,fig):
        post_corr(F,axs,H,fig)
        M=Fig(D+mat+'.svg').axes[4]; img=[a for a in M['arts'] if a['kind']=='image'][0]['arr']; y0,y1=M['bbox'][2],M['bbox'][3]
        for ax,key,lab in zip(axs,('Init','Ch','R'),('Init.','Ch.','Out.')):
            fr=[(t[0]-y0)/(y1-y0) for t in M['yt'] if t[1]==key][0]
            ia=ax.inset_axes([0.07,0.50,0.25,0.37]); ia.imshow(img,extent=(0,1,1,0),aspect='auto',interpolation='nearest')
            ia.axhline(fr,color='#e2001a',lw=0.6); ia.set_xticks([]); ia.yaxis.tick_right(); ia.set_yticks([fr],[lab],fontsize=6); ia.tick_params(length=0,pad=1)
            for sp in ia.spines.values(): sp.set_visible(False)
    return post
CT=[(r'^(Initiation|Choice|Outcome)$',r'\1 Time'),(r'^(CA1|PFC) (A|B)$',r'\2 \1'),(r'^Correlation Coefficient$','Correlation Coefficients'),(r'^(.*) < \.01$',r'\1 (p < .01)'),(r'^B - A PFC vs CA1$','Δ AB in CA1 > PFC')]
for n,m,mat in (('fig6c_policy_corr',2.4,'fig6d_policy_diag_1'),('ext5cd_policy_animal_1',4.5,'ext5cd_policy_animal_2')):
    go(n,scale=0.5,ms=m,cmap={'#c2abba':CA1,'#212053':PFC,'#ffc0cb':PINK,'#008000':'#218f53','#808080':GREY},legend='none',post=post_corr_inset(mat),xrot=r'Init|Ch|Out',text=CT)
def post_onebar(F,axs,H,fig):
    # as printed: the four matrices share one colour bar (same colour scale in all four)
    for ax in axs[1:-1:2]: ax.set_visible(False)
PT=[(r'^(CA1|PFC|HP) policy on (A|B) choice$',r'\1\nPolicy on \2 Choice\nCorrelations between Problems'),(r'^Init$','Initiation'),(r'^Ch$','Choice'),(r'^R$','Outcome'),(r'^-1$','Init. - 1s'),(r'^\+1$','Out. + 1s'),(r'^[-+]0\.5$','')]
for n in ('fig6d_policy_diag_1','ext5cd_policy_animal_2'):
    go(n,scale=0.42,text=PT,gap=(1.3,1.3),xrot=r'Init|Choice|Outcome|Out',ylabel=['','','','','','','','Correlation Coefficient'],post=post_onebar)
# ---------- Fig 1 I-L policy training
go('fig1ij_policy_training_1',scale=0.55,ylabel='Regression coefficient',cmap={'#212053':BGREEN},lw={'line':0.5})
go('fig1ij_policy_training_2',scale=0.55,ylabel='Regression coefficient',cmap={'#d1d3d5':BPINK},lw={'line':0.5})
go('fig1ij_policy_training_3',scale=0.5,post=ODD,annot=[(r'^0$','p < .01'),(r'^0\.01$','p = .013'),(r'^0\.(\d+)$',r'p = .\1')],text=[(r'^(.*):$',r'\1')],ylabel=['Regression coefficient','','',''],cmap=[{'#212053':BGREEN},{'#212053':BGREEN},{'#d1d3d5':BPINK},{'#d1d3d5':BPINK}],lw={'lines':1.5},gap=(1.1,1))
# ---------- Extended data
F1T=[(r'^Number of Out of Sequence Pokes$','Number of out-of-sequence\npokes per trial'),(r'^Number of Trials Till Threshold$','Number of trials\nto threshold'),(r'^Problem Number$','Problem'),(r'^Reversal Number$','Reversal number'),(r'^Probability of Choosing New Best Option$','Probability of choosing\nnew best option'),(r'^Trial # Before/After Reversal$','Trial number'),(r'^Mean Number of Out of Sequence Pokes$','Mean number of\nout-of-sequence pokes per trial'),(r'^Out of Sequence Pokes$','Mean number of\nout-of-sequence pokes per trial'),(r'^first 10 blocks$','First 10 blocks'),(r'^last 10 blocks$','Last 10 blocks'),(r'^block switch$','Reversal'),(r'^first 5 tasks$','First 5 problems'),(r'^last 5 tasks$','Last 5 problems'),(r'^problem (\d+)$',r'Problem \1')]
def post_byproblem(F,axs,H,fig):
    # as printed: one x label per problem (at its first reversal) instead of every reversal number, no legend
    ax=axs[0]; xt=list(ax.get_xticks()); lab=[t.get_text() for t in ax.get_xticklabels()]
    first=[x for x,l in zip(xt,lab) if l=='1']
    ax.set_xticks(first,[str(i+1) for i in range(len(first))],fontsize=7); ax.set_xlabel('Problem')
    if ax.get_legend(): ax.get_legend().remove()
F1C=dict(REG); F1C.update({'#ffc0cb':BPINK})
for n in ('fig1e_trials_reversal_1','fig1e_trials_reversal_2','fig1f_meta_learning','fig1gh_out_of_sequence_1','fig1gh_out_of_sequence_2','fig1gh_out_of_sequence_3'):
    c=dict(F1C)
    if n=='fig1f_meta_learning': c['#ffc0cb']='#d9777c'          # reversal line
    if n=='fig1gh_out_of_sequence_3': c['#ffc0cb']='#eeb0bb'     # last 5 problems
    go(n,scale=0.55 if '_2' not in n else 0.4,cmap=c,text=F1T,legend='right' if '_2' in n else 'best',lw={'line':1.3} if '_2' in n else None,post=EVEN if n=='fig1f_meta_learning' else post_solid if n in ('fig1e_trials_reversal_1','fig1gh_out_of_sequence_1') else post_byproblem if '_2' in n else None)
E2C=dict(REG); E2C.update({'#ffc0cb':CA1,'#800080':PFC})   # CA1 / PFC bars
E2J=[(r'^As$','A port'),(r'^Bs$','B port'),(r'^Other$','Other\nports'),(r'^Poke Identity$','Port identity'),(r'^Choice Probability$','Probability of first poke (%)'),(r'^A good, B forced$','B forced trials (A good)'),(r'^B good, A forced$','A forced trials (B good)')]
E4T=[(r'^cells with (Init|Ch) Peak in T(\d) firing in T(\d)$',r'Cells with \1. peak in layout \2,\nfiring in layout \3')]
for n in ('ext2ij_forced_1','ext2ij_forced_2','ext2ac_reversals_rec_1','ext2ac_reversals_rec_2','ext2ac_reversals_rec_3','ext2df_oos_rec_1','ext2df_oos_rec_2','ext2df_oos_rec_3','ext2df_oos_rec_4','ext4_remapping_1','ext4_remapping_5'):
    if 'forced' in n: go(n,scale=0.55,cmap=REG,post=post_pct,text=E2J)
    elif n=='ext4_remapping_5': go(n,scale=0.55,cmap=REG,text=E4T,gap=(1.3,1.15),xrot=r'Ch|Out')
    else: go(n,scale=0.55 if '_2' not in n else 0.4,cmap=E2C,text=F1T if n.startswith('ext2') else (),lw={'line':1.3} if '_2' in n else None,post=(post_byproblem if '_2' in n else post_solid) if n.startswith('ext2') else None)
# ---------- Ext 7 (DLC-aligned versions of 4C-E, 6B-E) and rest of Ext 4
RTT=[(r'^A vs B choice$','A vs B Choice'),(r'^Outcome at A vs B$','Outcome\nat A vs B'),(r'^Choice vs Initiation$','Choice vs\nInitiation'),(r'^Problem Specific A choice$','Problem Specific\nA Choice')]
go('ext7_dlc_rsa_1',scale=0.72,cmap=REG,ms=1.3,post=post_rsa1,xrot=r'Ch|Out',text=RTT,gap=(1.05,1))
go('ext7_dlc_rsa_2',scale=0.36,text=LAB+[(r'^HP\b','CA1')],gap=(1.25,1.35))
go('ext7_dlc_rsa_3',scale=0.42,noticks=True,text=RTT)
go('ext7_dlc_policy_glm',drop=r'^Prev Choice$',scale=0.45,ms=2.4,by_label=POL,legend='right',xrot=r'Init|Ch|Out',post=post_pol,text=[(r'^Reward x Choice$','Reward x Choice'),(r'^Reward$','Reward')])
go('ext7_dlc_policy_corr',scale=0.5,ms=2.4,cmap={'#c2abba':CA1,'#212053':PFC,'#ffc0cb':PINK,'#008000':'#218f53','#808080':GREY},legend='none',post=post_corr_inset('ext7_dlc_diagonal_1'),xrot=r'Init|Ch|Out',text=CT)
go('ext7_dlc_diagonal_1',scale=0.42,text=PT,gap=(1.3,1.3),xrot=r'Init|Choice|Outcome|Out',ylabel=['','','','','','','','Correlation Coefficient'],post=post_onebar)
go('ext7_dlc_diagonal_2',scale=HS,cmap=HI,text=HT,legend='none',lw={'lines':2.6},post=post_hist_perm,gap=(1.2,1))
for n in ('ext4_remapping_2','ext4_remapping_3','ext4_remapping_4','ext4_remapping_6'):
    go(n,scale=0.55,cmap=REG,**(dict(text=E4T,gap=(1.3,1.15),xrot=r'Ch|Out') if n=='ext4_remapping_6' else {}))
# ---------- Ext 2 d-g, all pokes (only_A_B=False)
for n in ('ext2dg_oos_all_1','ext2dg_oos_all_2','ext2dg_oos_all_3','ext2dg_oos_all_4'):
    if 'forced' in n: go(n,scale=0.55,cmap=REG,post=post_pct,text=E2J)
    elif n=='ext4_remapping_5': go(n,scale=0.55,cmap=REG,text=E4T,gap=(1.3,1.15),xrot=r'Ch|Out')
    else: go(n,scale=0.55 if '_2' not in n else 0.4,cmap=E2C,text=F1T if n.startswith('ext2') else (),lw={'line':1.3} if '_2' in n else None,post=(post_byproblem if '_2' in n else post_solid) if n.startswith('ext2') else None)
def post_dec3(F,axs,H,fig):
    # cell numbers: white on dark cells, ink on light cells (as in the generated figure)
    for ax in axs:
        for t in ax.texts:
            try: v=float(t.get_text())
            except ValueError: continue
            t.set_color('white' if v<60 else INK)
def post_posthoc(F,axs,H,fig):
    for i,ax in enumerate(axs):
        if i: ax.set_ylabel('')
        lg=ax.get_legend()
        if lg: lg.remove()
    axs[-1].set_xticklabels(['same\nport','same\nrole','different\nport'])
    for l in axs[-1].lines:
        if l.get_marker()=='o': l.set_markerfacecolor('#595a5c'); l.set_markersize(2.0); l.set_zorder(5)
    hs=[Line2D([],[],color=CA1,lw=1,label='CA1'),Line2D([],[],color=PFC,lw=1,label='PFC')]
    axs[-1].legend(handles=hs,loc='upper left',bbox_to_anchor=(1.03,1.0),handlelength=1.4,labelspacing=0.25,borderaxespad=0)
go('posthoc_port_over_time',scale=0.5,cmap={'#66c2a5':PFC,'#fc8d62':CA1,'#808080':GREY},solid=True,gap=(1.15,1.0),legend='none',post=post_posthoc,
   titles=['Same port','Same role','Different port (control)','Rewarded, choice to outcome'])
go('fig4fg_decode_1',scale=0.5,gap=(1.3,1.02),xrot=r'.',post=post_dec3)
# ---------- Ext 8 surprise, Fig 2 example neuron (added 28 Nov)
def post8(F,axs,H,fig):
    for i,ax in enumerate(axs):
        if ax.get_xlabel(): ax.set_xlabel('Trial before/after layout switch' if i==10 else '')
go('ext8_surprise',scale=0.45,gap=(1.12,1.22),post=post8,text=[(r'^(CA1|PFC) (A|B) T(\d) T(\d)$',r'\1 \2, layout \3 to \4')])
def post2(F,axs,H,fig): pass
go('fig2_neuron',scale=0.5,legend='best',legend_axes=[2],legend_kw=dict(loc='upper left',bbox_to_anchor=(1.3,-0.55),ncol=2,fontsize=7,borderaxespad=0),gap=(1.1,1),xrot=r'Init|Ch|Out',ylabel=['Firing rate (Hz)','','','Trial number']+['']*20,text=[(r'^Task Layout (\d)$',r'Problem \1'),(r'^A Rew$','A rewarded'),(r'^A NR$','A non-rewarded'),(r'^B Rew$','B rewarded'),(r'^B NR$','B non-rewarded'),(r'^task (\d)$',r'problem \1')])
