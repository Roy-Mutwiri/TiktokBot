# Verify face-aware body warp: face+neck stay untouched, only shoulders move,
# smooth neck transition. Applies a manual shoulder warp to the character image.
import os, sys
sys.path.insert(0, "engines")
import numpy as np, cv2
from body_motion import BodyMotionEngine, FRAME

char = "character_views/character.jpg"
if not os.path.exists(char):
    char = os.path.join("ai-face", "character.jpg")
img = cv2.resize(cv2.imread(char), (FRAME, FRAME))

b = BodyMotionEngine()
b._update_keep_line(img)            # find chin -> keep line
b._build_mask()
print("keep_y (below-chin line):", b._keep_y, "of", FRAME,
      f"({100*b._keep_y/FRAME:.0f}% down)")

# manual shoulder warp: tilt 5deg + sway 18px + slight scale, pivot at keep line
M = cv2.getRotationMatrix2D((FRAME/2.0, float(b._pivot_y)), 5.0, 1.03)
M[0, 2] += 18; M[1, 2] += 6
warped = cv2.warpAffine(img, M, (FRAME, FRAME), borderMode=cv2.BORDER_REPLICATE)
out = (img * (1.0 - b._mask) + warped * b._mask).astype(np.uint8)

# check: face region (above keep) must be IDENTICAL to source
diff_face = float(np.mean(np.abs(out[:b._keep_y].astype(int) - img[:b._keep_y].astype(int))))
print(f"face/neck region change above keep line: {diff_face:.3f} (should be ~0)")

# draw the keep line for the montage
vis = out.copy()
cv2.line(vis, (0, b._keep_y), (FRAME, b._keep_y), (0, 255, 255), 1)
cv2.putText(vis, "warped torso", (10, FRAME-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
cv2.putText(img, "original", (10, FRAME-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
cv2.imwrite("_body_check.jpg", np.hstack([img, vis]))
print("saved _body_check.jpg (left original, right warped+keepline)")
