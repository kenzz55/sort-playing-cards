import argparse
import cv2
import numpy as np
import sys
import os

# =============================
# Load Templates
# =============================
def load_ranks(path):
    ranks = []
    order = ['Ace','Two','Three','Four','Five','Six','Seven',
             'Eight','Nine','Ten','Jack','Queen','King']
    for name in order:
        img = cv2.imread(os.path.join(path, name + ".jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Rank template missing: {name}.jpg")
        ranks.append((name, img))
    return ranks


def load_suits(path):
    suits = []
    order = ['Spades','Hearts','Clubs','Diamonds']
    for name in order:
        img = cv2.imread(os.path.join(path, name + ".jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Suit template missing: {name}.jpg")
        suits.append((name, img))
    return suits

# =============================
# Utilities
# =============================
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def warp_card(img, pts):
    rect = order_points(pts)
    dst = np.array([[0, 0], [199, 0], [199, 299], [0, 299]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (200, 300))
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    return warp

# =============================
# Extract Rank & Suit
# =============================
def extract_rank_suit(warp):
    corner = warp[0:84, 0:32]
    zoom = cv2.resize(corner, (0, 0), fx=4, fy=4)

    white_level = int(zoom[15, 32 * 2])
    thresh_level = max(1, white_level - 30)
    _, th = cv2.threshold(zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)

    rank_region = th[20:185, 0:128]
    contours, _ = cv2.findContours(rank_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    rank = rank_region[y:y + h, x:x + w]
    rank = cv2.resize(rank, (70, 125))

    suit_region = th[186:336, 0:128]
    contours, _ = cv2.findContours(suit_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    suit = suit_region[y:y + h, x:x + w]
    suit = cv2.resize(suit, (70, 100))

    return rank, suit

# =============================
# Template Matching
# =============================
def match_template(img, templates):
    best_name = "Unknown"
    best_diff = 1e9
    for name, temp in templates:
        diff = cv2.absdiff(img, temp)
        score = diff.sum() / 255
        if score < best_diff:
            best_diff = score
            best_name = name
    return best_name, best_diff

# =============================
# Card Processing
# =============================
def process_cards(img, rank_templates, suit_templates):
    results = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000 or area > 1500000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
        if len(approx) != 4:
            continue

        box = np.array([p[0] for p in approx], dtype=np.int32)
        warp = warp_card(img, box)

        rank_img, suit_img = extract_rank_suit(warp)
        if rank_img is None or suit_img is None:
            continue

        rank_name, _ = match_template(rank_img, rank_templates)
        suit_name, _ = match_template(suit_img, suit_templates)
        results.append((rank_name, suit_name))

    return results

# =============================
# Sorting
# =============================
RANK_ORDER = {
    'Ace': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5,
    'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9, 'Ten': 10,
    'Jack': 11, 'Queen': 12, 'King': 13
}

SUIT_ORDER = {
    'Clubs': 1,
    'Diamonds': 2,
    'Hearts': 3,
    'Spades': 4
}

SUIT_MAP = {
    'Clubs': 'C',
    'Diamonds': 'D',
    'Hearts': 'H',
    'Spades': 'S'
}

# =============================
# Argparse / Main
# =============================
def parse_args():
    p = argparse.ArgumentParser("CV assignment runner")
    p.add_argument("--input", required=True, help="input image path")
    return p.parse_args()


def main():
    args = parse_args()
    img = cv2.imread(args.input)
    if img is None:
        return -1

    template_dir = "Card_Imgs"
    rank_templates = load_ranks(template_dir)
    suit_templates = load_suits(template_dir)

    results = process_cards(img, rank_templates, suit_templates)

    results.sort(key=lambda x: (RANK_ORDER[x[0]], SUIT_ORDER[x[1]]))

    output_tokens = []
    for r, s in results:
        suit_char = SUIT_MAP[s]
        rank_map = {
            "Ace": "A", "Two": "2", "Three": "3", "Four": "4", "Five": "5",
            "Six": "6", "Seven": "7", "Eight": "8", "Nine": "9", "Ten": "10",
            "Jack": "J", "Queen": "Q", "King": "K"
        }
        rank_char = rank_map[r]
        output_tokens.append(f"{suit_char}{rank_char}")

    print(" ".join(output_tokens))
    return 0

if __name__ == "__main__":
    sys.exit(main())
