use crate::consts::{
    DEFAULT_LEARNING_RATE, DEFAULT_NB_LAYERS, DEFAULT_NODES_PER_LAYER, NB_NEURAL_NETWORKS,
    NeuralNetwork, NeuralNetworkFloat,
};
use crate::neural_network::types::matrix::Matrix;
use rand::random_range;

pub struct NeuralNetworkParameters {
    pub learning_rate: NeuralNetworkFloat,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
}

pub trait NeuralNetworkTrait: Clone {
    fn get_output(&self, input: Matrix) -> NeuralNetworkFloat;
    fn get_outputs(&self, inputs: Vec<Matrix>) -> Vec<NeuralNetworkFloat>;

    fn change_learning_rate(&mut self, ratio: NeuralNetworkFloat);

    fn import(parameters: &NeuralNetworkParameters) -> Self;
    fn export(&self) -> NeuralNetworkParameters;

    fn train_once(&mut self, input: &Matrix, expected: NeuralNetworkFloat);
}

pub fn generate_neural_networks() -> Vec<NeuralNetwork> {
    let mut neural_networks = Vec::new();
    let mut nb_parameters = 0;
    for i in 0..DEFAULT_NB_LAYERS - 1 {
        // Biases are neglected, we only count weights
        nb_parameters += DEFAULT_NODES_PER_LAYER[i] * DEFAULT_NODES_PER_LAYER[i + 1];
    }
    println!("**************************************");
    println!("*** Generating new neural networks ***");
    println!("***  {} parameters  ***", nb_parameters);
    println!("**************************************");
    for _ in 0..NB_NEURAL_NETWORKS {
        let new_neural_network = NeuralNetwork::import(&generate_parameters(
            &DEFAULT_NODES_PER_LAYER,
            DEFAULT_LEARNING_RATE,
        ));
        neural_networks.push(new_neural_network);
    }
    neural_networks
}

pub fn get_nodes_per_layer(parameters: &NeuralNetworkParameters) -> Vec<usize> {
    let nb_layers = parameters.biases.len();
    let mut nodes_per_layer = vec![0; nb_layers];
    nodes_per_layer[0] = parameters.weights[1].width(); // No bias is applied to the first layer, so it cannot be used
    for i in 1..nb_layers {
        nodes_per_layer[i] = parameters.biases[i].height();
    }
    for i in 1..nb_layers {
        assert_eq!(nodes_per_layer[i - 1], parameters.weights[i].width());
        assert_eq!(nodes_per_layer[i], parameters.weights[i].height());
    }
    nodes_per_layer
}

fn generate_parameters(
    nodes_per_layer: &[usize],
    learning_rate: NeuralNetworkFloat,
) -> NeuralNetworkParameters {
    let weights = init_weights(nodes_per_layer);
    let biases = init_biases(nodes_per_layer);
    NeuralNetworkParameters {
        learning_rate,
        weights,
        biases,
    }
}

fn init_weights(nodes_per_layer: &[usize]) -> Vec<Matrix> {
    let mut weights = vec![Matrix::new(0, 0); nodes_per_layer.len()];
    for l in 1..nodes_per_layer.len() {
        let input_len = nodes_per_layer[l - 1];
        let output_len = nodes_per_layer[l];
        weights[l] = init_matrix(output_len, input_len);
    }
    weights
}

fn init_biases(nodes_per_layer: &[usize]) -> Vec<Matrix> {
    let mut biases = vec![Matrix::new(0, 0); nodes_per_layer.len()];
    for l in 1..nodes_per_layer.len() {
        let layer_len = nodes_per_layer[l];
        biases[l] = init_matrix(layer_len, 1);
    }
    biases
}

fn init_matrix(height: usize, width: usize) -> Matrix {
    let mut res = Matrix::new(height, width);
    for i in 0..res.height() {
        for j in 0..res.width() {
            res.set(i, j, random_range(-1. ..1.));
        }
    }
    res
}
