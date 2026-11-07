import numpy as np, cv2, os

def board_texture(inner=(8,5), px=100):
    cols, rows = inner[0]+1, inner[1]+1          # so o vuong
    tex = np.full(((rows+2)*px, (cols+2)*px), 255, np.uint8)  # vien trang 1 o
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                tex[(r+1)*px:(r+2)*px, (c+1)*px:(c+2)*px] = 0
    return tex

def render(K, rvec, tvec, size=(1280, 720), inner=(8,5), square=25.0, px=100):
    tex = board_texture(inner, px)
    S = np.array([[square/px, 0, -square], [0, square/px, -square], [0, 0, 1.0]])   # pixel texture -> mm the gioi (z=0)
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], np.asarray(tvec, float)]) @ S
    img = cv2.warpPerspective(tex, Hm, size, flags=cv2.INTER_LINEAR, borderValue=200)
    return img

def make_set(out_dir, n=20, seed=0, K=None, size=(1280, 720)):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1.0]]) if K is None else K
    paths = []
    for i in range(n):
        ang = np.deg2rad(rng.uniform(-35, 35, 3))
        rvec = cv2.Rodrigues(cv2.Rodrigues(np.array([np.pi, 0, 0.0]))[0] @ cv2.Rodrigues(ang)[0])[0].ravel()   # ban co huong ve camera
        tvec = np.array([rng.uniform(-80, 20), rng.uniform(-60, 30), rng.uniform(380, 650)])
        img = render(K, rvec, tvec, size)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        p = os.path.join(out_dir, f"img_{i:02d}.png"); cv2.imwrite(p, img); paths.append(p)
    return paths, K