from PIL import Image
im = Image.open(r"C:\Users\user\Documents\TiktokBot\_qa_turns.jpg")
w, h = im.size
cw, ch = w // 5, h // 2
labels = [["-55", "-40", "-25", "-10", "0"], ["+10", "+25", "+40", "+55", "0"]]
for r in range(2):
    for c in range(5):
        crop = im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
        crop = crop.resize((cw * 3, ch * 3))
        crop.save(rf"C:\Users\user\Documents\TiktokBot\_p_{r}_{c}_{labels[r][c]}.png")
print("done", im.size)
