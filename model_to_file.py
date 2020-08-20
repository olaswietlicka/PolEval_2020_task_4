from tensorflow.keras.models import model_from_json

def load_model_from_files(json_file, hdf5_file):
    json_file = open(json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(hdf5_file)
    return loaded_model

def save_model_to_file(model, type_of_model):
    model_file = 'model_' + type_of_model + '.json'
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
    print('Zapisano model sieci neuronowej do pliku {}'.format(model_file))

    weights_file = 'model_' + type_of_model + '.hdf5'
    model.save_weights(weights_file)
    print('Zapisano wagi sieci neuronowej do pliku {}'.format(weights_file))