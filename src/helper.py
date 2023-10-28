import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class ObjFileProcessor:
    def __init__(self, file_path: str):
        # Initialize the object with the path to the OBJ file.
        self.file_path = file_path
        self.vertices = None  # Store the vertices from the OBJ file.
        self.faces = None  # Store the faces from the OBJ file.

    def read_obj_file(self):
        # Read the OBJ file and extract vertices and faces.
        vertices = []  # Store 3D vertices.
        faces = []  # Store face information.

        with open(self.file_path, 'r') as obj_file:
            for line in obj_file:
                parts = line.strip().split()

                if not parts:
                    continue

                if parts[0] == 'v':
                    # Extract vertex coordinates (X, Y, Z) and convert them to floats.
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif parts[0] == 'f':
                    # Extract face information (vertex indices).
                    face = [int(vertex.split('/')[0]) for vertex in parts[1:]]
                    faces.append(face)

        self.vertices = np.array(vertices)
        self.faces = faces

    def split_and_label(self, split_z: float = 0.0):
        # Split vertices into upper and lower parts based on Z coordinate.
        upper_vertices = [vertex for vertex in self.vertices if vertex[2] >= split_z]
        lower_vertices = [vertex for vertex in self.vertices if vertex[2] < split_z]

        # Create labels for upper and lower vertices.
        upper_labels = np.ones(len(upper_vertices))  # 1 for upper vertices.
        lower_labels = np.zeros(len(lower_vertices))  # 0 for lower vertices.

        # Combine all vertices and labels.
        all_vertices = np.vstack((upper_vertices, lower_vertices))
        all_labels = np.hstack((upper_labels, lower_labels))

        return all_vertices, all_labels

    def create_model(self) -> keras.Model:
        # Create a simple neural network model for classification.
        model = keras.Sequential([
            layers.Input(shape=(3,)),           # Input layer with 3 features (X, Y, Z)
            layers.Dense(128, activation='relu'), # Dense hidden layer with 128 units
            layers.Dense(64, activation='relu'),  # Dense hidden layer with 64 units
            layers.Dense(1, activation='sigmoid') # Output layer with sigmoid activation
        ])

        # Compile the model with optimizer, loss function, and evaluation metric.
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100, batch_size: int = 32) -> keras.Model:
        # Create and train a classification model.
        model = self.create_model()
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        return model

    def plot_3d_classification(self, model: keras.Model, upper_vertices: np.ndarray, lower_vertices: np.ndarray):
        # Visualize the classification results in a 3D plot.
        fig = plt.figure(figsize=(12, 6))

        ax_upper = fig.add_subplot(121, projection='3d')

        ax_upper.scatter(upper_vertices[:, 0], upper_vertices[:, 1], upper_vertices[:, 2], c='red', marker='o',
                         label='Original Upper Side')
        # Use model.predict to obtain predictions for upper vertices.
        upper_predictions = model.predict(upper_vertices)
        ax_upper.scatter(upper_vertices[:, 0], upper_vertices[:, 1], upper_predictions, c='blue', marker='o',
                         label='Predicted Upper Side')
        ax_upper.set_xlabel('X')
        ax_upper.set_ylabel('Y')
        ax_upper.set_zlabel('Z')
        ax_upper.set_title('Predicted Upper Side')

        ax_lower = fig.add_subplot(122, projection='3d')

        ax_lower.scatter(lower_vertices[:, 0], lower_vertices[:, 1], lower_vertices[:, 2], c='red', marker='o',
                         label='Original Lower Side')
        # Use model.predict to obtain predictions for lower vertices.
        lower_predictions = model.predict(lower_vertices)
        ax_lower.scatter(lower_vertices[:, 0], lower_vertices[:, 1], lower_predictions, c='blue', marker='o',
                         label='Predicted Lower Side')
        ax_lower.set_xlabel('X')
        ax_lower.set_ylabel('Y')
        ax_lower.set_zlabel('Z')
        ax_lower.set_title('Predicted Lower Side')

        ax_upper.legend()
        ax_lower.legend()

        plt.tight_layout()
        plt.show()


