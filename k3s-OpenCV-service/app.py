#!/usr/bin/env python3
"""Independent camera, depth, localization, and streaming service."""

from __future__ import annotations

import html

import base_app

# Row 1: large live view beside the latest candy and hand scans stacked vertically.
# Row 2: the candy and hand scan metadata side by side. With several scan poses,
# the most recently captured candy and hand snapshots are shown.
PAGE_HTML = """<!doctype html>
<meta charset="utf-8">
<title>Piper Camera View</title>
<style>
body{margin:0;background:#111;color:#eee;font:15px sans-serif}
main{max-width:1600px;margin:auto;padding:12px}
h2{margin:4px 0 12px} h3{margin:0 0 6px;font-size:16px}
.top{display:grid;grid-template-columns:minmax(0,7fr) minmax(220px,3fr);gap:12px;align-items:start}
.live{display:block;width:100%;height:auto;border:1px solid #555}
.note{color:#00ff00;margin:6px 0 0;font-size:13px}
.scans{display:grid;gap:12px}
.details{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
section{background:#222;padding:8px}
.frame{background:#000;aspect-ratio:4/3}
.frame img{display:block;width:100%;height:100%;object-fit:contain}
.frame img:not([src]){visibility:hidden}
pre{white-space:pre-wrap;font-size:12px;margin:0;max-height:480px;overflow:auto}
@media(max-width:900px){.top,.details{grid-template-columns:1fr}}
</style>
<main><h2>Camera View</h2>
<div class="top">
<div><img class="live" src="/stream.mjpg" alt="Live RealSense stream">
<p class="note">Green wireframe: candy workspace sent by demo.py. It appears only
while demo.py is supplying fresh Piper pose telemetry.</p></div>
<div class="scans">
<section><h3>Candy scan</h3><div class="frame"><img id="candyshot" alt="Latest candy scan"></div></section>
<section><h3>Hand scan</h3><div class="frame"><img id="handshot" alt="Latest hand scan"></div></section>
</div>
</div>
<div class="details">
<section><h3>Candy scan details</h3><pre id="candymeta">Not captured</pre></section>
<section><h3>Hand scan details</h3><pre id="handmeta">Not captured</pre></section>
</div></main>
<script>
const shown={candy:null,hand:null};
function latest(snapshots){
  let best=null;
  for(const [slot,value] of Object.entries(snapshots||{})){
    if(!best||String(value.captured_at||'')>String(best.value.captured_at||'')) best={slot,value};
  }
  return best;
}
function show(kind,imagePath,snapshots){
  const best=latest(snapshots);
  if(!best) return;
  const key=best.slot+'|'+(best.value.captured_at||'');
  if(shown[kind]===key) return;
  shown[kind]=key;
  document.getElementById(kind+'shot').src=imagePath+best.slot+'.jpg?t='+Date.now();
  document.getElementById(kind+'meta').textContent=JSON.stringify(best.value,null,2);
}
async function refresh(){
  try{
    const r=await fetch('/api/snapshots',{cache:'no-store'});
    show('candy','/snapshots/',(await r.json()).snapshots);
  }catch(error){
    document.getElementById('candymeta').textContent='Camera service unavailable: '+error;
  }
  try{
    const r=await fetch('/api/hand-snapshots',{cache:'no-store'});
    show('hand','/hand-snapshots/',(await r.json()).snapshots);
  }catch(error){
    document.getElementById('handmeta').textContent='Hand service unavailable: '+error;
  }
}
setInterval(refresh,1000); refresh();
</script>"""


def page_html() -> bytes:
    page = PAGE_HTML.encode("utf-8")
    if base_app.env_bool("CAMERA_PREVIEW_ONLY", False):
        page = page.replace(
            b"<main>",
            b"<main><p style='color:#ffcc00'>PREVIEW ONLY: camera images are available. "
            b"Depth/XYZ localization and workspace overlays are disabled until this "
            b"camera's calibration is accepted.</p>",
        )
    return page


base_app.page_html = page_html


class VisionApplication(base_app.VisionApplication):
    detection_enabled = False
    service_name = "opencv-vision"

    def __init__(self) -> None:
        super().__init__()
        self.camera.show_status_banner = False

    def dashboard_html(self) -> bytes:
        camera_name = html.escape(self.camera.name or "Camera identity unavailable")
        service_name = html.escape(self.service_name)
        title = f"Camera View — {camera_name} | {service_name}".encode("utf-8")
        return page_html().replace(b"<h2>Camera View</h2>", b"<h2>" + title + b"</h2>", 1)


create_server = base_app.create_server


def main() -> int:
    port = base_app.env_int("PORT", 8094)
    app = VisionApplication()
    try:
        app.start()
        server = create_server(app, port)
        print(f"OpenCV camera UI listening on 0.0.0.0:{port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping OpenCV vision service", flush=True)
    finally:
        app.camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
