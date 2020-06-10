from PIL import Image, ImageDraw


def get_triangle(points, dim=(320, 180)):
    triangle = Image.new('1', dim)
    drawer = ImageDraw.Draw(triangle)
    drawer.polygon(points, fill=1)
    # triangle.save('/home/mir/Desktop/1.png')
    return triangle
