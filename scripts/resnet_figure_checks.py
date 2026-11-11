"""Stability and agreement of the two ResNet50 figures (LIME vs Grad-CAM)."""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
ROOT = Path.home() / "Documents" / "TEBOL"; sys.path.insert(0, str(ROOT / "src"))
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
qf = load(str(ROOT/"scripts/qualitative_figures.py"), "qf"); sg = load(str(ROOT/"scripts/survey_gradcam.py"), "sg")
B = __import__("tebol.bbox", fromlist=["x"])
from lime import lime_image
from torchvision import models, transforms
cats = models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
model = sg.load_model(); norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
def predict(batch):
    out = []
    for chunk in np.array_split(np.asarray(batch, dtype=np.uint8), max(1, len(batch)//50)):
        t = torch.stack([norm(torch.from_numpy(a.astype(np.float32)/255).permute(2,0,1)) for a in chunk])
        with torch.no_grad(): out.append(torch.softmax(model(t),1).numpy())
    return np.concatenate(out)
rows = []
for name,(cls,stem,inet) in qf.GRADCAM_FIGS.items():
    target = [i for i,c in enumerate(cats) if c.lower()==inet][0]
    img_f = sg.read_image(qf.find_raw(stem)); img = np.uint8(np.round(img_f*255))
    cam = sg.grad_cam(model, norm(torch.from_numpy(img_f).permute(2,0,1))[None], [target])[0]
    W=[]; segs=None
    for sd in (qf.SEED, 11, 22):
        e = lime_image.LimeImageExplainer(random_state=sd).explain_instance(img, predict, labels=(target,), hide_color=0, num_samples=qf.SAMPLES_RESNET, random_seed=sd, batch_size=50)
        segs = e.segments; ids = np.unique(segs); heat = qf.heat_from(e, target)
        W.append(np.array([heat[segs==s][0] for s in ids]))
    camseg = np.array([cam[segs==s].mean() for s in ids])
    rho = [spearmanr(W[0],W[k]).correlation for k in (1,2)]
    top = [set(ids[np.argsort(-w)[:3]]) for w in W]
    jac = [len(top[0]&top[k])/len(top[0]|top[k]) for k in (1,2)]
    row = {"figure":name,"stem":stem,"seed_spearman":round(float(np.mean(rho)),2),"seed_top3_jaccard":round(float(np.mean(jac)),2),
           "lime_vs_gradcam_spearman":round(float(spearmanr(W[0],camseg).correlation),2)}
    xml = list((ROOT/"data/annotations").glob(f"*/*/{stem}.xml"))
    if xml:
        boxes,Wd,Hd = B.truth_boxes(xml[0]); mask = np.zeros((224,224),bool)
        for x0,y0,x1,y1 in boxes: mask[int(y0*224/Hd):int(np.ceil(y1*224/Hd)), int(x0*224/Wd):int(np.ceil(x1*224/Wd))] = True
        pos = np.clip(qf.heat_from(e,target),0,None); camp = cam/cam.sum()
        row.update(box_area_share=round(float(mask.mean()),2), lime_pos_in_box=round(float(pos[mask].sum()/pos.sum()),2), gradcam_mass_in_box=round(float(camp[mask].sum()),2))
    rows.append(row); print(row, flush=True)
pd.DataFrame(rows).to_csv(ROOT/"results/metrics/resnet_figure_checks.csv", index=False)
