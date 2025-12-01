import onnx
from onnxconverter_common import float16

src = "glaucoma_unet.onnx"
dst = "glaucoma_unet_fp16.onnx"

print(f"Loading {src} ...")
model = onnx.load(src)

print("Converting to float16 ...")
model_fp16 = float16.convert_float_to_float16(model)

print(f"Saving to {dst} ...")
onnx.save(model_fp16, dst)
print("Done.")
