"""H3 Image style-library batch renderer. Resumable: rerun to continue.

Renders every style in data/styles.json for one base prompt via RenderBox
(official MiniMax H3 image t2i graph: image UNET int8 + qwen3vl CLIP + image VAE,
KSampler euler/simple 6 steps cfg 1, LLM caption-expansion branch NOT used so the
style text goes in verbatim). Fixed seed per base for cross-style comparability.

Usage: python render_batch.py <base_prompt_id>
Outputs: server folder h3stylelib/<base_id> -> zipped once at the end ->
images/<base_id>/<style_id>.webp + data/generation_manifest.json (Krea-site schema).
"""
import io, json, os, sys, time, urllib.request, zipfile
from pathlib import Path
from PIL import Image

COMFY = "http://192.168.100.100:8188"
ROOT = Path(__file__).resolve().parents[1]
SEED = 123456789
STEPS = 6
WEBP_Q = 82
WEBP_MAX_SIDE = 2048  # site display size; originals live on the Comfy server


def api(path, data=None, method="POST", raw=False):
    req = urllib.request.Request(COMFY + path,
                                 data=json.dumps(data).encode() if data is not None else None,
                                 method=method)
    req.add_header("Content-Type", "application/json")
    resp = urllib.request.urlopen(req, timeout=120)
    body = resp.read()
    if raw:
        return body
    return json.loads(body) if body.strip() else {}


def graph(prompt, seed, prefix, w, h):
    return {
        "23:1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_image_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "23:2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "23:3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_image_vae_fp16.safetensors"}},
        "23:5": {"class_type": "MiniMaxH3ImageConditioning", "inputs": {"prompt": prompt, "width": w, "height": h, "clip": ["23:2", 0], "vae": ["23:3", 0]}},
        "23:6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["23:5", 0]}},
        "23:7": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": STEPS, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
                                                       "model": ["23:1", 0], "positive": ["23:5", 0], "negative": ["23:6", 0], "latent_image": ["23:5", 1]}},
        "23:8": {"class_type": "VAEDecode", "inputs": {"samples": ["23:7", 0], "vae": ["23:3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["23:8", 0]}},
    }


def main():
    base_id = sys.argv[1] if len(sys.argv) > 1 else "portrait"
    styles = json.load(open(ROOT / "data/styles.json", encoding="utf-8"))["styles"]
    bp = json.load(open(ROOT / "data/base_prompts.json", encoding="utf-8"))
    base = next(p for p in bp["prompts"] if p["id"] == base_id)
    tpl = bp["prompt_template"]

    state_path = ROOT / "scripts" / f"state_{base_id}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": {}, "failed": {}}

    targets = [("base", "base")] + [(s["id"], s["id"]) for s in styles]
    queue = [(sid, key) for sid, key in targets if sid not in state["done"] and state["failed"].get(sid, 0) < 2]
    print(f"base={base_id} total={len(targets)} todo={len(queue)}", flush=True)

    t_start = time.time()
    for i, (style_id, _) in enumerate(queue):
        if style_id == "base":
            prompt = tpl.replace("{base_prompt}", base["prompt"]).replace("{style_name}", "").replace("{style_descriptor}", "").strip()
            # base card: subject only, no style clause
            prompt = base["prompt"]
        else:
            st = next(s for s in styles if s["id"] == style_id)
            prompt = tpl.format(base_prompt=base["prompt"], style_name=st["name"], style_descriptor=st["descriptor"])
        prefix = f"h3stylelib/{base_id}/{style_id}"
        g = graph(prompt, SEED, prefix, base["width"], base["height"])
        try:
            pid = api("/api/prompt", {"prompt": g, "client_id": "hermes", "number": 90, "priority": 90})["prompt_id"]
        except Exception as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            print(f"[{i+1}/{len(queue)}] {style_id} SUBMIT-ERR {e} {body}", flush=True)
            state["failed"][style_id] = state["failed"].get(style_id, 0) + 1
            state_path.write_text(json.dumps(state))
            continue
        t0 = time.time()
        while True:
            try:
                h = api(f"/history/{pid}", method="GET")
            except Exception as e:
                print(f"  poll err {e}", flush=True); time.sleep(10); continue
            if pid in h:
                e = h[pid]["status"]
                if not e.get("completed"):
                    print(f"[{i+1}/{len(queue)}] {style_id} RENDER-FAILED", flush=True)
                    state["failed"][style_id] = state["failed"].get(style_id, 0) + 1
                else:
                    img = h[pid]["outputs"]["9"]["images"][0]
                    state["done"][style_id] = {"filename": img["filename"], "subfolder": img["subfolder"], "secs": round(time.time() - t0)}
                    el = time.time() - t_start
                    print(f"[{i+1}/{len(queue)}] {style_id} ok {state['done'][style_id]['secs']}s | batch {el/60:.1f}min eta {(len(queue)-i-1)*el/(i+1)/60:.0f}min", flush=True)
                break
            time.sleep(6)
        state_path.write_text(json.dumps(state))

    # download whole output folder as zip, convert to webp
    print("downloading outputs (per-file)...", flush=True)
    out_dir = ROOT / "images" / base_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_imgs = {}
    for style_id, info in state["done"].items():
        dest = out_dir / f"{style_id}.webp"
        if dest.exists() and dest.stat().st_size > 1000:
            manifest_imgs[f"{base_id}/{style_id}"] = {"base_prompt_id": base_id, "style_id": style_id,
                "kind": "base" if style_id == "base" else "style", "status": "complete",
                "path": f"images/{base_id}/{style_id}.webp", "seed": SEED,
                "width": base["width"], "height": base["height"]}
            continue
        url = f"{COMFY}/api/view?filename={urllib.request.quote(info['filename'])}&subfolder={urllib.request.quote(info['subfolder'])}&type=output"
        png = urllib.request.urlopen(url, timeout=300).read()
        if len(png) < 1000:
            print(f"  EMPTY download {style_id}", flush=True); continue
        im = Image.open(io.BytesIO(png))
        if max(im.size) > WEBP_MAX_SIDE:
            r = WEBP_MAX_SIDE / max(im.size)
            im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
        im.save(dest, "WEBP", quality=WEBP_Q)
        manifest_imgs[f"{base_id}/{style_id}"] = {"base_prompt_id": base_id, "style_id": style_id,
            "kind": "base" if style_id == "base" else "style", "status": "complete",
            "path": f"images/{base_id}/{style_id}.webp", "seed": SEED,
            "width": base["width"], "height": base["height"]}
        print(f"  webp {style_id} {dest.stat().st_size//1024}KB", flush=True)

    mpath = ROOT / "data/generation_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"schema_version": 1, "updated_at": None, "images": {}}
    manifest["images"].update(manifest_imgs)
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"DONE base={base_id} done={len(state['done'])} failed={len(state['failed'])}", flush=True)
    api("/free", {"unload_models": True, "free_memory": True})  # 200 empty body
    print("VRAM freed", flush=True)


if __name__ == "__main__":
    main()
