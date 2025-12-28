use crate::NB_LAYERS;
use crate::players::neural_network::neural_networks_types::matrix::Matrix;
use rand::random_range;
use std::fs;
use std::fs::File;
use std::io::Write;

pub trait NeuralNetworkTrait: Clone + Send + Sync {
    fn get_output(&self, input: Matrix) -> f64;
    fn get_outputs(&self, inputs: Vec<Matrix>) -> Vec<f64>;

    fn change_learning_rate(&mut self, ratio: f64);

    fn import(parameters: (Vec<Matrix>, Vec<Matrix>, f64)) -> Self;
    fn export(&self) -> (Vec<Matrix>, Vec<Matrix>, f64);

    fn train_once(&mut self, input: Matrix, expected: f64);
}

pub fn export_parameters(parameters: (Vec<Matrix>, Vec<Matrix>, f64), path: &str) {
    let (weights, biases, learning_rate) = parameters;
    let mut file = File::create(path).unwrap();
    let mut content = String::new();
    content += &learning_rate.to_string();
    for l in 0..NB_LAYERS {
        content += "\n\n";
        content += &export_matrix(&weights[l]);
        content += "\n\n";
        content += &export_matrix(&biases[l]);
    }
    file.write_all(content.as_bytes()).unwrap();
}

pub fn import_parameters(path: &str) -> (Vec<Matrix>, Vec<Matrix>, f64) {
    let content = fs::read_to_string(path).unwrap();
    assert_eq!(content.split("\n\n").count(), 2 * NB_LAYERS + 1);
    let mut blocs = content.split("\n\n");
    let learning_rate = blocs.next().unwrap().parse().unwrap();
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    for (id, matrix) in blocs.enumerate() {
        if id % 2 == 0 {
            weights.push(import_matrix(matrix));
        } else {
            biases.push(import_matrix(matrix));
        }
    }
    (weights, biases, learning_rate)
}

fn export_matrix(matrix: &Matrix) -> String {
    let mut res = String::new();
    for x in 0..matrix.height() {
        for y in 0..matrix.width() {
            res += &matrix.get(x, y).to_string();
            res += " ";
        }
        res.pop(); // Removes trailing space
        res += "\n";
    }
    res.pop(); // Removes trailing new line
    res
}

fn import_matrix(text: &str) -> Matrix {
    if text.is_empty() {
        return Matrix::new(0, 0);
    }
    let height = text.lines().count();
    let width = text.lines().next().unwrap().split(" ").count();
    let mut res = Matrix::new(height, width);
    for (x, line) in text.lines().enumerate() {
        for (y, value) in line.split(" ").enumerate() {
            res.set(x, y, value.parse::<f64>().unwrap());
        }
    }
    res
}

pub fn generate_parameters(
    nodes_per_layer: [usize; NB_LAYERS],
    learning_rate: f64,
) -> (Vec<Matrix>, Vec<Matrix>, f64) {
    let weights = init_weights(&nodes_per_layer);
    let biases = init_biases(&nodes_per_layer);
    (weights, biases, learning_rate)
}

fn init_weights(nodes_per_layer: &[usize; NB_LAYERS]) -> Vec<Matrix> {
    let mut weights = vec![Matrix::new(0, 0); NB_LAYERS];
    for l in 1..NB_LAYERS {
        let input_len = nodes_per_layer[l - 1];
        let output_len = nodes_per_layer[l];
        let mut weight = Matrix::new(output_len, input_len);
        for i in 0..weight.height() {
            for j in 0..weight.width() {
                weight.set(i, j, random_range(-1. ..1.));
            }
        }
        weights[l] = weight;
    }
    weights
}

fn init_biases(nodes_per_layer: &[usize; NB_LAYERS]) -> Vec<Matrix> {
    let mut biases = vec![Matrix::new(0, 0); NB_LAYERS];
    for l in 1..NB_LAYERS {
        let layer_len = nodes_per_layer[l];
        let mut bias = Matrix::new(layer_len, 1);
        for i in 0..bias.height() {
            for j in 0..bias.width() {
                bias.set(i, j, random_range(-1. ..1.));
            }
        }
        biases[l] = bias;
    }
    biases
}
