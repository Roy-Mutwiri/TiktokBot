import os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.getcwd(), "engines"))
VID = r"C:/Users/user/Downloads/Telegram Desktop/video_2026-06-10_14-45-12.mp4"
cap = cv2.VideoCapture(VID); cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ok, fr = cap.read(); cap.release()
fr = cv2.resize(fr, (512,512))
print("cv2 threads:", cv2.getNumThreads())

def bench(label, fn, n=60):
    fn()  # warm
    best=1e9; tot=0
    for _ in range(n):
        s=time.perf_counter(); fn(); e=time.perf_counter()-s
        tot+=e; best=min(best,e)
    print(f"  {label:34s} avg {tot/n*1000:6.2f} best {best*1000:6.2f}")

# Break apply_color_grade into pieces
def cg_astype():
    f = fr.astype(np.float32)/255.0
def cg_exposure():
    f = fr.astype(np.float32)/255.0; f = np.clip(f*1.12,0,1)
def cg_power():
    f = fr.astype(np.float32)/255.0; f = np.clip(f*1.12,0,1); f=np.power(f,0.86)
def cg_full():
    f = fr.astype(np.float32)/255.0
    f = np.clip(f*1.12,0,1)
    f = np.power(f,0.86)
    luma=(0.114*f[:,:,0]+0.587*f[:,:,1]+0.299*f[:,:,2])[:,:,None]
    hi=np.clip((luma-0.5)*2.0,0,1)
    f[:,:,2]=np.clip(f[:,:,2]+hi[:,:,0]*0.030+0.015,0,1)
    f[:,:,1]=np.clip(f[:,:,1]+hi[:,:,0]*0.008+0.004,0,1)
    f=np.clip((f-0.5)*1.05+0.5,0,1)
    return (f*255).astype(np.uint8)
print("=== color_grade breakdown ===")
bench("astype/255", cg_astype)
bench("+ exposure clip", cg_exposure)
bench("+ np.power", cg_power)
bench("full color_grade", cg_full)

print("=== grain breakdown ===")
def grain_randn(): n=(np.random.randn(512,512,1)*8.0).astype(np.float32)
def grain_full():
    n=(np.random.randn(512,512,1)*8.0).astype(np.float32)
    return np.clip(fr.astype(np.float32)+n,0,255).astype(np.uint8)
bench("np.random.randn(512,512,1)", grain_randn)
bench("full grain", grain_full)

# cheaper grain alt: randint uint8 noise
def grain_fast():
    n=np.random.randint(-8,9,(512,512,1),dtype=np.int16).astype(np.int16)
    return np.clip(fr.astype(np.int16)+n,0,255).astype(np.uint8)
bench("grain via randint int16", grain_fast)
