from PIL import Image, ImageDraw, ImageFont

def draw(image, text, font_path, font_size=40, bg_opacity=153, text_opacity=204):

        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font = ImageFont.truetype(font_path, font_size)

        bbox = draw.textbbox((0, 0), text, font=font, anchor="lt")
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle(((0, 0), (text_width, text_height)), fill=(0, 0, 0, bg_opacity))

        draw.text((0, 0), text, font=font, fill=(255, 255, 255, text_opacity), anchor="lt")

        result = Image.alpha_composite(image, overlay)
        return result
