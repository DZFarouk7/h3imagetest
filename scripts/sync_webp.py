
"""Incremental sync: download finished renders + rebuild manifest anytime."""
import io, json, time, urllib.request, urllib.parse
from pathlib import Path
from PIL import Image
COMFY="http://192.168.100.100:8188"
ROOT=Path(r"C:/Users/Legion/h3-style-library")
BASE="portrait"; WEBP_MAX_SIDE=2048; WEBP_Q=82
state=json.loads((ROOT/"scripts"/f"state_{BASE}.json").read_text())
out_dir=ROOT/"images"/BASE; out_dir.mkdir(parents=True,exist_ok=True)
imgs={}
n=0
for sid,info in state["done"].items():
    dest=out_dir/f"{sid}.webp"
    if dest.exists() and dest.stat().st_size>1000:
        imgs[f"{BASE}/{sid}"]={"base_prompt_id":BASE,"style_id":sid,"kind":"base" if sid=="base" else "style","status":"complete","path":f"images/{BASE}/{sid}.webp","seed":123456789,"width":3072,"height":5472}
        continue
    url=f"{COMFY}/api/view?filename={urllib.parse.quote(info['filename'])}&subfolder={urllib.parse.quote(info['subfolder'])}&type=output"
    png=urllib.request.urlopen(url,timeout=300).read()
    if len(png)<1000: print("EMPTY",sid); continue
    im=Image.open(io.BytesIO(png))
    if max(im.size)>WEBP_MAX_SIDE:
        r=WEBP_MAX_SIDE/max(im.size); im=im.resize((round(im.width*r),round(im.height*r)),Image.LANCZOS)
    im.save(dest,"WEBP",quality=WEBP_Q); n+=1
    imgs[f"{BASE}/{sid}"]={"base_prompt_id":BASE,"style_id":sid,"kind":"base" if sid=="base" else "style","status":"complete","path":f"images/{BASE}/{sid}.webp","seed":123456789,"width":3072,"height":5472}
mpath=ROOT/"data"/"generation_manifest.json"
mpath.write_text(json.dumps({"schema_version":1,"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"images":imgs},ensure_ascii=False,indent=1),encoding="utf-8")
print(f"manifest={len(imgs)} new_webp={n}")
