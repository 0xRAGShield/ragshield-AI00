from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

source = Path("data/images/manar_abdelbaky_prescription2023.jpeg")
out = Path("outputs/image_enhancement_tests")
out.mkdir(parents=True, exist_ok=True)

im = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
gray = ImageOps.grayscale(im)

# 1. تحسين التباين
contrast = ImageOps.autocontrast(gray, cutoff=1)
contrast.save(out / "01_contrast.png")

# 2. تحسين التباين مع Sharpening خفيف
sharp = ImageEnhance.Sharpness(contrast).enhance(1.8)
sharp.save(out / "02_sharpen.png")

# 3. تقليل التشويش مع الحفاظ على الحواف قدر الإمكان
denoised = gray.filter(ImageFilter.MedianFilter(size=3))
denoised = ImageOps.autocontrast(denoised, cutoff=1)
denoised.save(out / "03_denoise_contrast.png")

# 4. نسخة أبيض وأسود بعتبة قابلة للتعديل
threshold = contrast.point(lambda p: 255 if p > 170 else 0)
threshold.save(out / "04_threshold.png")

print(f"Original size: {im.size}")
print(f"Saved variants in: {out.resolve()}")
