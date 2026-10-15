import io, zipfile, urllib.request, re, collections, sys, os
URL="https://huggingface.co/datasets/ShotaroFukunaga/xview_coco_crop_1008/resolve/main/xview_coco_crop_1008.zip"
class H(io.RawIOBase):
    def __init__(s,u):
        s.u=u; r=urllib.request.urlopen(urllib.request.Request(u,method="HEAD")); s.n=int(r.headers["Content-Length"]); s.p=0
    def seekable(s): return True
    def readable(s): return True
    def seek(s,o,w=0): s.p={0:o,1:s.p+o,2:s.n+o}[w]; return s.p
    def tell(s): return s.p
    def read(s,k=-1):
        if k<0: k=s.n-s.p
        if k==0 or s.p>=s.n: return b""
        e=min(s.n,s.p+k)-1
        d=urllib.request.urlopen(urllib.request.Request(s.u,headers={"Range":f"bytes={s.p}-{e}"})).read(); s.p+=len(d); return d
    def readinto(s,b): d=s.read(len(b)); b[:len(d)]=d; return len(d)
z=zipfile.ZipFile(io.BufferedReader(H(URL),buffer_size=1<<20))
ids=collections.defaultdict(set)
for n in z.namelist():
    m=re.search(r'(train_images|val_images)/(\d+)_y',n)
    if m: ids[m.group(1)].add(m.group(2))
root=os.path.expanduser(sys.argv[1])
ours={sp:{p[:-4] for p in os.listdir(f"{root}/{sp}/images")} for sp in ("train","val")}
for sp in ours:
    for k in ("train_images","val_images"):
        print(sp, len(ours[sp]), "in ckpt", k, len(ours[sp]&ids[k]))
    print(sp,"in neither", len(ours[sp]-ids["train_images"]-ids["val_images"]))
held=sorted(ours["train"]-ids["train_images"])
open("heldout_train_ids.txt","w").write("\n".join(held)); print("held-out train ids:", len(held))
