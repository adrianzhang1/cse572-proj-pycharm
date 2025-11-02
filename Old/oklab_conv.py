import numpy as np


def srgb_to_linear(c):
    """Convert an sRGB channel (0–1) to linear RGB"""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def srgb_to_oklab(rgb):
    # Normalize to 0–1 range
    r,g,b = rgb
    r, g, b = [x / 255.0 for x in (r, g, b)]

    # Convert to linear RGB
    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)
    # r_lin, g_lin, b_lin = r,g,b

    # Linear RGB → LMS
    l = 0.4122214708 * r_lin + 0.5363325363 * g_lin + 0.0514459929 * b_lin
    m = 0.2119034982 * r_lin + 0.6806995451 * g_lin + 0.1073969566 * b_lin
    s = 0.0883024619 * r_lin + 0.2817188376 * g_lin + 0.6299787005 * b_lin

    # Non-linear transformation
    l_, m_, s_ = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)

    # LMS → OKLab
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return L, a, b
def linear_to_srgb(c):
    """Convert a linear RGB channel (0–1) to sRGB"""
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055

def oklab_to_srgb(Lab):
    # OKLab → LMS
    L,a,b = Lab
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    # LMS³ → linear RGB
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3

    r_lin = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    # linear RGB → sRGB
    r = linear_to_srgb(r_lin)
    g = linear_to_srgb(g_lin)
    b = linear_to_srgb(b_lin)

    # Clamp and convert to 0–255
    r = int(round(max(0, min(1, r)) * 255))
    g = int(round(max(0, min(1, g)) * 255))
    b = int(round(max(0, min(1, b)) * 255))

    return r, g, b

def oklab_to_oklch(lab):
    l, a, b = lab
    c = np.sqrt(a ** 2 + b ** 2)
    h_rad = np.arctan2(b, a)
    # h_deg = np.degrees(h_rad) % 360
    return l, c, h_rad

# print(srgb_to_oklab([255, 254, 254]))
# print(oklab_to_srgb(srgb_to_oklab([255, 255, 255])))
# print(oklab_to_srgb((0.5,0.1,0.1)))