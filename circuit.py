import pennylane as qml
import torch

class QuantumCircuit:
    """
    The Quantum Brain: A Variational Quantum Circuit (VQC).
    
    Architecture:
    1. Embedding: Encodes classical features (angles) into Qubit states.
    2. Ansatz: Entangling layers to process information.
    3. Measurement: Measure Qubits to get outputs.
    """
    def __init__(self, n_qubits=4, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Define the QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # 1. Encoding
            # AngleEmbedding rotates the qubit state based on input feature.
            # We map 'n_qubits' features to 'n_qubits' wires.
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            
            # 2. Ansatz (The "Neural Network" part of the circuit)
            # StronglyEntanglingLayers applies rotations and CNOTs (entanglement).
            # This creates complex correlations between inputs.
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
            # 3. Measurement
            # We measure the Expectation value of PauliZ for each qubit.
            # Output range: [-1, 1]
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
            
        self.qnode = circuit
        
        # Weight Shapes for StrongEntanglingLayers: (n_layers, n_qubits, 3)
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}

    def get_torch_layer(self):
        """
        Returns a PennyLane TorchLayer that can be inserted into a PyTorch model.
        Output dimension = n_qubits
        """
        return qml.qnn.TorchLayer(self.qnode, self.weight_shapes)

if __name__ == "__main__":
    # Test the circuit
    n_qubits = 4
    qc = QuantumCircuit(n_qubits=n_qubits)
    q_layer = qc.get_torch_layer()
    
    # Dummy Input: Batch of 1, 4 features
    dummy_input = torch.rand(1, n_qubits)
    output = q_layer(dummy_input)
    
    print(f"Input: {dummy_input}")
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    print("Quantum Circuit Initialized Successfully.")
