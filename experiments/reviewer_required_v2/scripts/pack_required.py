from __future__ import annotations
import hashlib, io, json, os, tarfile
from pathlib import Path
from common import CFG, REPORT, WORK
def main():
    if not (REPORT/"PROTOCOL_SEAL.json").exists(): raise RuntimeError("Protocol is not sealed")
    out=Path(os.environ.get("RESULT_ARCHIVE",f"/root/autodl-tmp/{CFG['protocol_id']}_results.tar.gz")).expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    included=["reports/"+CFG["protocol_id"],"results/"+CFG["protocol_id"],"groups","data","logs"]
    manifest={"protocol_id":CFG["protocol_id"],"workspace":str(WORK),"included":included,"note":"source assets and model weights are not copied into result archive"}
    with tarfile.open(out,"w:gz") as a:
        for rel in included:
            src=WORK/rel
            if src.exists(): a.add(src,arcname=CFG["protocol_id"]+"/"+rel,recursive=True)
        raw=(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n").encode(); info=tarfile.TarInfo(CFG["protocol_id"]+"/RESULT_ARCHIVE_MANIFEST.json"); info.size=len(raw); a.addfile(info,io.BytesIO(raw))
    digest=hashlib.sha256(out.read_bytes()).hexdigest(); out.with_name(out.name+".sha256").write_text(f"{digest}  {out.name}\n",encoding="utf-8",newline="\n"); print(f"PACKED: {out}\nSHA256: {digest}")
if __name__=="__main__": main()
