"""Rebuild a matplotlib-exported SVG in the paper's Illustrator style.
Geometry is read straight from the SVG (exact vector coordinates); only style, labels and layout scale change."""
import xml.etree.ElementTree as ET, re, io, base64, logging
import numpy as np
import matplotlib; matplotlib.use('Agg')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Patch
from matplotlib.lines import Line2D
from PIL import Image
NS='{http://www.w3.org/2000/svg}'; XL='{http://www.w3.org/1999/xlink}href'
CA1='#c2acbb'; PFC='#252552'; INK='#1d1d1b'; GREY='#9d9d9d'
plt.rcParams.update({'font.family':['Arial','Liberation Sans'],'font.size':8,'axes.titlesize':8,'axes.titleweight':'bold',
 'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':0.5,'xtick.major.width':0.5,'ytick.major.width':0.5,
 'xtick.major.size':2,'ytick.major.size':2,'xtick.major.pad':2,'ytick.major.pad':2,'axes.labelpad':2,
 'axes.edgecolor':INK,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
 'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False,'legend.fontsize':7,'mathtext.default':'regular'})
TEXT_DEFAULT=[(r'^HP\b','CA1'),(r'\bHP\b','CA1'),(r'^Time in Trial \(s\)$','Time in trial'),(r'^Init$','Init.'),(r'^Ch$','Ch.'),(r'^R$','Out.'),(r'^C$','Ch.'),
 (r'^Reward x Choice$','Outcome x Choice'),(r'^Reward$','Outcome'),(r'^neuron #$','Neuron #'),(r'^Permuted differences$','Permuted Differences')]
def style(el):
    d={}
    for kv in (el.get('style') or '').split(';'):
        if ':' in kv: k,v=kv.split(':',1); d[k.strip()]=v.strip()
    return d
def parse_path(d):
    toks=re.findall(r'[MLCQz]|-?[\d.]+(?:e-?\d+)?',d); v=[];c=[];i=0;cmd=None
    n={'M':1,'L':1,'C':3,'Q':2}; code={'M':Path.MOVETO,'L':Path.LINETO,'C':Path.CURVE4,'Q':Path.CURVE3}
    while i<len(toks):
        t=toks[i]
        if t in 'MLCQ': cmd=t; i+=1; continue
        if t=='z': v.append((0,0)); c.append(Path.CLOSEPOLY); i+=1; continue
        for k in range(n[cmd]):
            v.append((float(toks[i]),float(toks[i+1]))); c.append(code[cmd]); i+=2
        if cmd=='M': cmd='L'
    return Path(np.array(v).reshape(-1,2),c) if v else None
def tidy(s):
    s=s.replace('−','-').strip()
    if re.fullmatch(r'-?\d+\.\d+',s):
        s=s.rstrip('0').rstrip('.')
        if s in('-0',''): s='0'
    return s
def text_of(g):
    for ch in g.iter():
        if ch.tag is ET.Comment: return __import__("html").unescape(ch.text.strip())
    return ''
def text_pos(g):
    for ch in g.iter(NS+'g'):
        tr=ch.get('transform')
        if tr and 'translate' in tr:
            m=re.search(r'translate\(([-\d.e]+)[ ,]+([-\d.e]+)\)',tr); rot=re.search(r'rotate\(([-\d.e]+)',tr); sc=re.search(r'scale\(([-\d.e]+)',tr)
            return float(m.group(1)),float(m.group(2)),(-float(rot.group(1)) if rot else 0),(float(sc.group(1))*100 if sc else 10)
    return None
class Fig:
    def __init__(self,path):
        self.root=ET.parse(path,parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()
        self.W=float(self.root.get('width')[:-2]); self.H=float(self.root.get('height')[:-2])
        self.defs={}
        for el in self.root.iter():
            if el.get('id') and el.tag in (NS+'path',NS+'image'): self.defs[el.get('id')]=el
        fig=[g for g in self.root if g.get('id')=='figure_1'][0]
        self.axes=[self.parse_axes(g) for g in fig if g.get('id','').startswith('axes')]
        self.figtexts=[(text_of(g),text_pos(g)) for g in fig if g.get('id','').startswith('text')]
    def uses(self,g):
        out=[]
        for u in g.iter(NS+'use'):
            ref=self.defs.get(u.get(XL)[1:])
            if ref is None or u.get('x') is None: continue
            st=style(ref); st.update(style(u)); out.append((float(u.get('x')),float(u.get('y')),parse_path(ref.get('d')),st))
        return out
    def paths(self,g):
        return [(parse_path(p.get('d')),style(p)) for p in g.iter(NS+'path') if p.get('id') is None and p.get('d')]
    def artist(self,g):
        gid=g.get('id') or ''
        kind=re.sub(r'_?\d+$','',gid)
        imgs=list(g.iter(NS+'image'))
        if imgs:
            im=imgs[0]; data=base64.b64decode(im.get(XL).split(',',1)[1]); arr=np.array(Image.open(io.BytesIO(data)))
            w=float(im.get('width')); h=float(im.get('height')); x=float(im.get('x')); y=float(im.get('y')); tr=im.get('transform') or ''
            m=re.search(r'translate\(([-\d.e]+)[ ,]+([-\d.e]+)\)',tr)
            # matplotlib writes: transform="scale(1 -1) translate(0 -h)" with y = -ybottom
            y_top=-y if 'scale(1 -1)' in tr else y
            if 'scale(1 -1)' in tr: arr=arr[::-1]
            return dict(kind='image',arr=arr,ext=(x,x+w,y_top+h,y_top))
        us=self.uses(g); ps=self.paths(g)
        if kind=='line2d':
            if us: return dict(kind='markers',pts=us)
            if ps: return dict(kind='line',path=ps[0][0],st=ps[0][1])
        if kind in('PathCollection',): return dict(kind='markers',pts=us,scatter=True)
        if kind=='FillBetweenPolyCollection' or (us and not ps): return dict(kind='polys',pts=us)
        if kind=='LineCollection': return dict(kind='lines',paths=ps)
        if kind=='patch' and ps: return dict(kind='patch',path=ps[0][0],st=ps[0][1])
        if kind=='text': return dict(kind='text',s=text_of(g),pos=text_pos(g))
        return None
    def parse_axes(self,g):
        ax=dict(arts=[],xt=[],yt=[],xlabel=None,ylabel=None,legend=None,spines=set(),id=g.get('id'))
        first=True
        for ch in g:
            gid=ch.get('id') or ''
            if gid.startswith('matplotlib.axis'):
                for t in ch:
                    tid=t.get('id') or ''
                    if tid.startswith(('xtick','ytick')):
                        u=self.uses(t); txt=[c for c in t if (c.get('id') or '').startswith('text')]
                        lab=text_of(txt[0]) if txt else ''; pos=text_pos(txt[0]) if txt else None
                        if u: (ax['xt'] if tid.startswith('x') else ax['yt']).append((u[0][0] if tid.startswith('x') else u[0][1],lab,pos))
                    elif tid.startswith('text'):
                        s=text_of(t); pos=text_pos(t)
                        if pos and abs(pos[2])>45: ax['ylabel']=(s,pos)
                        else: ax['xlabel']=(s,pos)
                continue
            if gid.startswith('legend'):
                ent=[];cur=[]
                for c in list(ch)[1:]:
                    cid=c.get('id') or ''
                    if cid.startswith('text'): ent.append((cur,text_of(c))); cur=[]
                    else:
                        a=self.artist(c)
                        if a: cur.append(a)
                ax['legend']=ent; continue
            a=self.artist(ch)
            if a is None: continue
            if a['kind']=='patch' and first:
                first=False; v=a['path'].vertices[:-1]; ax['bbox']=(v[:,0].min(),v[:,0].max(),v[:,1].min(),v[:,1].max()); continue
            ax['arts'].append(a)
        x0,x1,y0,y1=ax['bbox']; keep=[]
        for a in ax['arts']:
            if a['kind']=='patch' and a['st'].get('fill')=='none' and len(a['path'].vertices)<=5:
                v=a['path'].vertices; v=v[a['path'].codes!=Path.CLOSEPOLY]
                if len(v)==2:
                    if abs(v[0,0]-v[1,0])<1e-6: ax['spines'].add('left' if abs(v[0,0]-x0)<1 else 'right')
                    else: ax['spines'].add('bottom' if abs(v[0,1]-y1)<1 else 'top')
                    continue
                if len(v)==4: ax['spines']|={'left','right','top','bottom'}; continue
            keep.append(a)
        ax['arts']=keep
        return ax
def render(fig,out,scale=0.45,cmap=None,text=(),skip_text=(),legend='best',lw=None,ms=1.3,gap=(1.0,1.0),titles=None,post=None,
           xrot=None,keep_colors=False,by_label=None,noticks=False,annot=(),solid=False,scatter_color=None,xlabel=None,ylabel=None,legend_kw=None,legend_axes=None,figsize_pad=(0,0),label_map=None,fmt=('svg','pdf','png'),aximg_interp='nearest'):
    cmaps=cmap if isinstance(cmap,list) else [cmap]*len(fig.axes)
    cmaps=[{k.lower():v for k,v in (c or {}).items()} for c in cmaps]
    if by_label:
        for A in fig.axes:
            for arts,lab in (A['legend'] or []):
                for pat,newc in by_label.items():
                    if arts and re.search(pat,tidy(lab)):
                        a=arts[0]; st=a.get('st') or (a['paths'][0][1] if a['kind']=='lines' else a['pts'][0][3])
                        oc=st.get('stroke') if a['kind'] in('line','lines') else st.get('fill')
                        for c in cmaps: c.setdefault(oc.lower(),newc)
                        break
    cmap={}
    def col(c):
        if c is None or c=='none': return 'none'
        c=c.lower(); 
        if c in cmap: return cmap[c]
        return INK if c=='#000000' else c
    rules=list(text)+TEXT_DEFAULT
    def T(s):
        s=tidy(s)
        for a,b in rules:
            if re.search(a,s): s=re.sub(a,b,s); break
        return s
    def LW(w,kind='line'):
        w=float(w or 1)
        if lw and kind in lw: return lw[kind]
        return 1.0 if w>=1.4 else max(0.5,w*0.7)
    W,H=fig.W*scale*gap[0]/72,fig.H*scale*gap[1]/72
    F=plt.figure(figsize=(W+figsize_pad[0],H+figsize_pad[1])); FW,FH=F.get_size_inches(); axs=[]; handles_all=[]
    for i,A in enumerate(fig.axes):
        x0,x1,y0,y1=A['bbox']; w=(x1-x0)*scale/72; h=(y1-y0)*scale/72
        cx=(x0+x1)/2/fig.W*W; cy=(1-(y0+y1)/2/fig.H)*H+figsize_pad[1]
        cmap.clear(); cmap.update(cmaps[i])
        ax=F.add_axes([(cx-w/2)/FW,(cy-h/2)/FH,w/FW,h/FH]); axs.append(ax)
        ax.set_xlim(x0,x1); ax.set_ylim(y1,y0)
        for s in ('left','right','top','bottom'): ax.spines[s].set_visible(s in A['spines'])
        for a in A['arts']:
            k=a['kind']
            if k=='image': ax.imshow(a['arr'],extent=a['ext'],aspect='auto',interpolation=aximg_interp,zorder=0); ax.set_xlim(x0,x1); ax.set_ylim(y1,y0)
            elif k=='line':
                st=a['st']; v=a['path'].vertices; dash=st.get('stroke-dasharray')
                ax.plot(v[:,0],v[:,1],color=col(st.get('stroke')),lw=LW(st.get('stroke-width')),ls='--' if dash else '-',alpha=float(st.get('opacity',st.get('stroke-opacity',1))),solid_capstyle='round',dash_capstyle='butt')
            elif k=='lines':
                for p,st in a['paths']:
                    v=p.vertices; ax.plot(v[:,0],v[:,1],color=col(st.get('stroke')),lw=LW(st.get('stroke-width'),'lines'),alpha=float(st.get('opacity',st.get('stroke-opacity',1))),ls='--' if st.get('stroke-dasharray') else '-')
            elif k=='markers':
                for (x,y,p,st) in a['pts']:
                    e=p.get_extents(); size=max(e.width,e.height)
                    if len(p.vertices)<=3:  # line-like marker
                        ax.plot(p.vertices[:,0]+x,p.vertices[:,1]+y,color=col(st.get('stroke')),lw=0.5)
                    else:
                        fc=col(st.get('fill')); ec=col(st.get('stroke'))
                        ax.plot([x],[y],'o',ms=max(size*scale*ms,0.8) if not a.get('scatter') else max(size*scale*0.8,1.5),mfc=(scatter_color if (scatter_color and a.get('scatter')) else fc if fc!='none' else ec),mec='none',alpha=float(st.get('opacity',st.get('fill-opacity',1))),clip_on=False)
            elif k=='polys':
                for (x,y,p,st) in a['pts']:
                    pp=Path(p.vertices+[x,y],p.codes); ax.add_patch(PathPatch(pp,fc=col(st.get('fill','#000000')),ec='none',alpha=float(st.get('opacity',st.get('fill-opacity',1))),lw=0))
            elif k=='patch':
                st=a['st']; ec=col(st.get('stroke')); ax.add_patch(PathPatch(a['path'],fc=col(st.get('fill','#000000')),ec=ec,lw=0 if ec=='none' else LW(st.get('stroke-width'),'patch'),alpha=1 if solid else float(st.get('opacity',st.get('fill-opacity',1)))))
            elif k=='text' and a['pos']:
                s=a['s']; px,py,rot,fs=a['pos']
                if any(re.search(r,s) for r in skip_text): continue
                if py<y0+2 and rot==0:
                    ax.set_title(T(s) if not titles else titles[i] if titles[i] is not None else T(s),pad=4)
                else:
                    s2=tidy(s); hit=False
                    for a_,b_ in annot:
                        if re.search(a_,s2): s2=re.sub(a_,b_,s2); hit=True; break
                    ax.text(px,py,s2 if hit else T(s),rotation=rot,fontsize=7,va='baseline',ha='left')
        # ticks
        right=any(t[2] and t[2][0]>x1 for t in A['yt'])
        top=any(t[2] and t[2][1]<y0 for t in A['xt'])
        ax.set_xticks([t[0] for t in A['xt']]); ax.set_xticklabels([T(t[1]) for t in A['xt']])
        ax.set_yticks([t[0] for t in A['yt']]); ax.set_yticklabels([T(t[1]) for t in A['yt']])
        if right: ax.yaxis.tick_right(); ax.yaxis.set_label_position('right')
        if top: ax.xaxis.tick_top()
        rot=[t[2][2] for t in A['xt'] if t[2]]
        if rot and abs(rot[0])>45: plt.setp(ax.get_xticklabels(),rotation=90,ha='center')
        if xrot: 
            for l in ax.get_xticklabels():
                if re.search(xrot,l.get_text()): l.set_rotation(45); l.set_ha('right'); l.set_rotation_mode('anchor')
        if A['xlabel']: ax.set_xlabel(T(A['xlabel'][0]))
        if A['ylabel']: ax.set_ylabel(T(A['ylabel'][0]))
        if xlabel is not None: ax.set_xlabel(xlabel if isinstance(xlabel,str) else xlabel[i])
        if ylabel is not None: ax.set_ylabel(ylabel if isinstance(ylabel,str) else ylabel[i])
        if titles and titles[i] is not None: ax.set_title(titles[i],pad=4)
        if noticks: ax.set_xticks([]); ax.set_yticks([])
        if A['legend']:
            hs=[]
            for arts,lab in A['legend']:
                a=arts[0] if arts else None; lab=T(lab)
                if a is None: continue
                if a['kind']=='line': st=a['st']; hs.append(Line2D([],[],color=col(st.get('stroke')),lw=LW(st.get('stroke-width')),ls='--' if st.get('stroke-dasharray') else '-',label=lab))
                elif a['kind']=='lines': st=a['paths'][0][1]; hs.append(Line2D([],[],color=col(st.get('stroke')),lw=LW(st.get('stroke-width'),'lines'),label=lab))
                elif a['kind']=='markers':
                    x,y,p,st=a['pts'][0]; fc=col(st.get('fill')); hs.append(Line2D([],[],ls='',marker='o',ms=max(p.get_extents().width*scale*ms,1.2),mfc=fc,mec='none',label=lab))
                elif a['kind']=='patch': st=a['st']; hs.append(Patch(fc=col(st.get('fill','#000000')),ec='none',alpha=float(st.get('opacity',1)),label=lab))
            handles_all.append((i,hs))
            if legend=='right' and (legend_axes is None or i in legend_axes): ax.legend(handles=hs,loc='upper left',bbox_to_anchor=(1.02,1.0),handlelength=1.4,labelspacing=0.2,borderaxespad=0,**(legend_kw or {}))
            if legend=='best' and (legend_axes is None or i in legend_axes): ax.legend(handles=hs,**{**dict(loc='best',handlelength=1.4,labelspacing=0.2,borderaxespad=0.2),**(legend_kw or {})})
    for s,pos in fig.figtexts:
        if pos: F.text(pos[0]/fig.W,1-pos[1]/fig.H,T(s),fontsize=8,fontweight='bold')
    if post: post(F,axs,handles_all,fig)
    for e in fmt: F.savefig(f'{out}.{e}',dpi=300,bbox_inches='tight',pad_inches=0.05)
    plt.close(F); return axs
