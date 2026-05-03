import tflite_runtime.interpreter as tflite

interp = tflite.Interpreter(model_path="checkpoints/keyword_spotting_awq.tflite")
interp.allocate_tensors()

for t in interp.get_tensor_details():
    print(t['index'], t['name'], t['shape'])