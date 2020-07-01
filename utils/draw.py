from PIL import Image, ImageDraw


def get_triangle(points, dim=(320, 180)):
    triangle = Image.new('1', dim)
    drawer = ImageDraw.Draw(triangle)
    # x0 y0 x1 y1 x2 y2
    # radius = 5
    # x_p0_tl = points[0] - radius
    # x_p0_br = points[0] + radius
    # y_p0_tl = points[1] - radius
    # y_p0_br = points[1] + radius
    # x_p1_tl = points[2] - radius
    # x_p1_br = points[2] + radius
    # y_p1_tl = points[3] - radius
    # y_p1_br = points[3] + radius
    # x_p2_tl = points[4] - radius
    # x_p2_br = points[4] + radius
    # y_p2_tl = points[5] - radius
    # y_p2_br = points[5] + radius
    # drawer.ellipse([x_p0_tl, y_p0_tl, x_p0_br, y_p0_br], fill=1)
    # drawer.ellipse([x_p1_tl, y_p1_tl, x_p1_br, y_p1_br], fill=1)
    # drawer.ellipse([x_p2_tl, y_p2_tl, x_p2_br, y_p2_br], fill=1)
    drawer.polygon(points, fill=1)
    # triangle.save('/home/mir/Desktop/1.png')
    return triangle


# get_triangle([87,  94,  60, 136, 144, 152])
