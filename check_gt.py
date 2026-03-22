import cv2, numpy as np
from pathlib import Path

TEST_DIR = Path("runs/ood_test")

gt = {
    "alien_frame_00530_png.rf.d584836cfbc516a6c67f422c7d2d35d3.jpg": [(124,35,180,91),(114,568,170,624)],
    "alien_frame_00545_png.rf.b1330f4bdd39ee2f4e894f39c023fec2.jpg": [(99,442,155,498),(42,40,98,96)],
    "alien_frame_00999_png.rf.6bfd635f12778b091ed35405acf0bec6.jpg": [(248,527,304,583),(568,439,624,495)],
    "alien_frame_01036_png.rf.92a937b89ba341e52bc3a2142e3eea17.jpg": [(294,16,350,72),(54,480,110,536)],
    "alien_frame_01106_png.rf.c23ead82f3a98818dc205f3702353406.jpg": [(306,91,362,147),(185,556,241,612)],
    "alien_frame_01295_png.rf.2f80d02f7b55708de4e3d7f9f194f7b7.jpg": [(483,398,539,454),(244,42,300,98)],
    "alien_frame_01523_png.rf.792bf1294b8127f97ef332a6de2f1760.jpg": [(227,521,283,577),(415,479,471,535)],
    "alien_frame_01720_png.rf.8a56d5d93de05700ab5dab15f3190fea.jpg": [(561,279,617,335),(448,418,504,474)],
    "alien_frame_01723_png.rf.907fc4295dc04b0dc53700928823a022.jpg": [(151,531,207,587),(442,75,498,131)],
    "alien_frame_01785_png.rf.0e2541fc7f3fbec5ceb9728b07cd608f.jpg": [(489,551,545,607),(267,21,323,77)],
}

print("Checking GT box center pixel (yellow = HSV H:15-35, S>100):")
print()
all_ok = True
for img_name, boxes in gt.items():
    img = cv2.imread(str(TEST_DIR / img_name))
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pixel_bgr = img[cy, cx]
        px  = np.uint8([[pixel_bgr]])
        hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
        is_yellow = (15 <= int(hsv[0]) <= 35) and int(hsv[1]) > 100
        status = "YELLOW OK" if is_yellow else "*** NOT YELLOW - GT WRONG ***"
        if not is_yellow:
            all_ok = False
        print(f"  {img_name[:42]}  box{i+1} ({cx},{cy})  BGR={list(pixel_bgr)}  HSV={list(hsv)}  {status}")

print()
print("GT reconstruction:", "ALL CORRECT" if all_ok else "SOME BOXES ARE WRONG - GT mismatch!")
