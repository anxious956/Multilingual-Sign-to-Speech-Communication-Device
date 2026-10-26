"""Fig. 11 of the proposal: drawings reproduced from three of the patents reviewed in Section 8.

python docs/figures/make_section8_figures.py

Downloads the drawing sheets from Google Patents (U.S. patent drawings are public records), crops the figure of
interest, rotates sideways sheets upright and puts them side by side with (a), (b), (c) labels.
"""
import io
import os
import subprocess

from PIL import Image, ImageChops, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
SHEETS = {
    # Sign-Speak, US 2022/0327961 A1 (granted as US 12,518,653 B2), drawing sheet 4: FIG. 7 is the top half
    "signspeak": ("https://patentimages.storage.googleapis.com/b4/0b/4b/57edc5233654e8/"
                  "US20220327961A1-20221013-D00004.png", (0, 0, 1.0, 0.47), 0),
    # Avodah, US 10,489,639 B2, sheet 4: FIG. 4, drawn sideways
    "avodah": ("https://patentimages.storage.googleapis.com/1f/90/cf/55db2ef673a179/US10489639-20191126-D00004.png",
               (0, 0, 1.0, 1.0), -90),
    # Sorenson, US 12,131,586 B2, sheet 3: FIG. 3, drawn sideways
    "sorenson": ("https://patentimages.storage.googleapis.com/b0/48/e0/dcdbafe1519cf8/US12131586-20241029-D00003.png", (0, 0, 1.0, 1.0), -90),
}


def fetch(url):
    # curl rather than urllib: the image server answers urllib requests with 403
    data = subprocess.run(["curl", "-sSfL", url], capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(data)).convert("L")


def trim(im, pad=12):
    """Crop to the drawing, leaving a small white margin."""
    bg = Image.new("L", im.size, 255)
    box = ImageChops.difference(im, bg).point(lambda v: 255 if v > 40 else 0).getbbox()
    im = im.crop(box) if box else im
    out = Image.new("L", (im.width + 2 * pad, im.height + 2 * pad), 255)
    out.paste(im, (pad, pad))
    return out


def panel(key):
    url, (x0, y0, x1, y1), rot = SHEETS[key]
    im = fetch(url)
    im = im.crop((int(x0 * im.width), int(y0 * im.height), int(x1 * im.width), int(y1 * im.height)))
    if rot:
        im = im.rotate(rot, expand=True, fillcolor=255)
    return trim(im)


def main():
    """(a) and (b) side by side on the first row, (c) centred on the second, so that each drawing stays legible."""
    a, b, c = (panel(k) for k in ("signspeak", "avodah", "sorenson"))
    W, gap, label = 2600, 80, 70
    h1 = round((W - gap) / (a.width / a.height + b.width / b.height))
    a, b = (i.resize((round(i.width * h1 / i.height), h1), Image.LANCZOS) for i in (a, b))
    h2 = min(round(c.height * W / c.width), round(1.15 * h1))
    c = c.resize((round(c.width * h2 / c.height), h2), Image.LANCZOS)
    out = Image.new("L", (W, label + h1 + gap + label + h2), 255)
    d = ImageDraw.Draw(out)
    font = ImageFont.truetype(FONT, 56)
    spots = [(a, 0, label), (b, a.width + gap, label), (c, (W - c.width) // 2, label + h1 + gap + label)]
    for tag, (im, x, y) in zip("abc", spots):
        out.paste(im, (x, y))
        d.text((x + 6, y - label + 4), f"({tag})", fill=0, font=font)
    out.save(os.path.join(HERE, "fig_patent_drawings.png"))
    print("wrote fig_patent_drawings.png", out.size)


if __name__ == "__main__":
    main()
