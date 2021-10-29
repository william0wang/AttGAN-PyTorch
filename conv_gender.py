import coremltools
from coremltools.models.neural_network import quantization_utils

# folder = 'cnn_age_gender_models_and_data.0.0.2'

# coreml_model = coremltools.converters.caffe.convert(
#     ('gender_net.caffemodel', 'deploy_gender.prototxt'),
#     image_input_names = 'data',
#     class_labels = 'genders.txt'
# )
# coreml_model.author = 'William Wang'
# coreml_model.license = 'Unknown'
# coreml_model.short_description = 'Gender Classification using Convolutional Neural Networks'
# coreml_model.input_description['data'] = 'An image with a face.'
# coreml_model.output_description['prob'] = 'The probabilities for each gender, for the given input.'
# coreml_model.output_description['classLabel'] = 'The most likely gender, for the given input.'

# model_fp8 = quantization_utils.quantize_weights(coreml_model, nbits=8)

# model_fp8.save('gender8.mlmodel')


mlmodel =  coremltools.models.MLModel('gender.mlmodel')
model_fp2 = quantization_utils.quantize_weights(mlmodel, nbits=4)
model_fp2.save('gender4.mlmodel')
