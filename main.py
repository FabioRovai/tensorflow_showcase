import json

from sklearn.model_selection import train_test_split

from src.helper import ObjFileProcessor

if __name__ == "__main__":
    # Open a JSON file located at './params/config.json' for reading
    with open('./params/config.json', 'r') as json_file:
        # Load the data from the JSON file into the 'loaded_data' variable
        loaded_data = json.load(json_file)

    # Retrieve the value associated with the key "file_path" from the loaded JSON data
    obj_file_path = loaded_data["file_path"]

    # Print the loaded file path to the console
    print("Loaded file path:", obj_file_path)

    # Create an instance of the 'ObjFileProcessor' class with the specified file path
    obj_processor = ObjFileProcessor(obj_file_path)

    # Read and process the data from the OBJ file using the 'read_obj_file' method
    obj_processor.read_obj_file()

    # Split and label the vertices from the OBJ file based on the specified criteria (split_z=0.0)
    all_vertices, all_labels = obj_processor.split_and_label(split_z=0.0)

    # Split the data into training and testing sets using an 80/20 split and a random seed of 42
    X_train, X_test, y_train, y_test = train_test_split(all_vertices, all_labels, test_size=0.2, random_state=42)

    # Train a classifier using the training data with a specified number of epochs (10 in this case)
    model = obj_processor.train_classifier(X_train, y_train, epochs=10)

    # Evaluate the trained model's accuracy using the testing data
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Print the classifier's accuracy to the console
    print("Classifier Accuracy:", accuracy)

    # Generate a 3D classification plot using the trained model and the data points
    # from both the training and testing sets
    obj_processor.plot_3d_classification(model, all_vertices[:len(X_train)], all_vertices[len(X_train):])



