from predict import predict

def predict_images():
    input_folder = './image/'
    model_path = './model/vision/'
    model_name = 'model_vision'
    output_folder = model_path
    result = predict(input_folder, model_path, model_name, output_folder)
    return result

#result = predict_images()
#print(result)
