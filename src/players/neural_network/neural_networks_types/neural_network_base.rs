use crate::players::neural_network::neural_networks_types::matrix::Matrix;
use crate::players::neural_network::neural_networks_types::neural_network::NeuralNetworkTrait;
use crate::{NB_LAYERS, NODES_PER_LAYER};
use ndarray::{Array2, arr2};

#[derive(Clone)]
pub struct NeuralNetworkBase {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    learning_rate: f64,
}

impl NeuralNetworkTrait for NeuralNetworkBase {
    fn get_output(&self, input: Matrix) -> f64 {
        let mut output = array2_of_matrix(&input);
        for l in 1..NB_LAYERS {
            output = get_layer_output(&output, &self.weights[l], &self.biases[l]);
        }
        output[[0, 0]]
    }

    fn get_outputs(&self, inputs: Vec<Matrix>) -> Vec<f64> {
        let mut output = Array2::from_shape_fn((NODES_PER_LAYER[0], inputs.len()), |(i, j)| {
            inputs[j].get(i, 0)
        });
        for l in 1..NB_LAYERS {
            output = get_layer_output(&output, &self.weights[l], &self.biases[l]);
        }
        let mut res = Vec::new();
        for i in 0..inputs.len() {
            res.push(output[[0, i]]);
        }
        res
    }

    fn change_learning_rate(&mut self, ratio: f64) {
        self.learning_rate *= ratio;
    }

    fn import(parameters: (Vec<Matrix>, Vec<Matrix>, f64)) -> Self {
        let (weights, biases, learning_rate) = parameters;
        let weights = weights.iter().map(array2_of_matrix).collect::<Vec<_>>();
        let biases = biases.iter().map(array2_of_matrix).collect::<Vec<_>>();
        Self {
            weights,
            biases,
            learning_rate,
        }
    }

    fn export(&self) -> (Vec<Matrix>, Vec<Matrix>, f64) {
        let weights = self
            .weights
            .iter()
            .map(matrix_of_array2)
            .collect::<Vec<_>>();
        let biases = self.biases.iter().map(matrix_of_array2).collect::<Vec<_>>();
        (weights, biases, self.learning_rate)
    }

    fn train_once(&mut self, input: Matrix, expected: f64) {
        let values = get_values(
            array2_of_matrix(&input),
            &self.weights,
            &self.biases,
            NB_LAYERS,
        );
        let grad_over_z = get_grad_over_z(&values, expected, &self.weights, NB_LAYERS);
        let grad_over_weights = get_grad_over_weights(&values, &grad_over_z, NB_LAYERS);
        let grad_over_biases = get_grad_over_biases(&grad_over_z);
        for l in 1..NB_LAYERS {
            self.weights[l] = &self.weights[l] - self.learning_rate * &grad_over_weights[l];
            self.biases[l] = &self.biases[l] - self.learning_rate * &grad_over_biases[l];
        }
    }
}

fn float_sigmoid(x: f64) -> f64 {
    1. / (1. + f64::exp(-x))
}

fn sigmoid(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(float_sigmoid)
}

fn get_layer_output(input: &Array2<f64>, weight: &Array2<f64>, bias: &Array2<f64>) -> Array2<f64> {
    sigmoid(&(weight.dot(input) + bias))
}

fn get_values(
    input: Array2<f64>,
    weights: &Vec<Array2<f64>>,
    biases: &Vec<Array2<f64>>,
    nb_layers: usize,
) -> Vec<Array2<f64>> {
    let mut values = vec![Array2::default((0, 0)); nb_layers];
    values[0] = input;
    for l in 1..nb_layers {
        values[l] = get_layer_output(&values[l - 1], &weights[l], &biases[l]);
    }
    values
}

fn get_grad_over_z(
    values: &Vec<Array2<f64>>,
    expected: f64,
    weights: &Vec<Array2<f64>>,
    nb_layers: usize,
) -> Vec<Array2<f64>> {
    let mut grad_over_z = vec![Array2::default((0, 0)); nb_layers];
    for l in (1..nb_layers).rev() {
        let grad_over_values = if l == nb_layers - 1 {
            let guessed = values[values.len() - 1][[0, 0]];
            arr2(&[[((1. - expected) / (1. - guessed)) - expected / guessed]])
        } else {
            weights[l + 1].t().dot(&grad_over_z[l + 1])
        };
        grad_over_z[l] = &grad_over_values * &values[l].mapv(|x| x * (1. - x));
    }
    grad_over_z
}

fn get_grad_over_weights(
    values: &Vec<Array2<f64>>,
    grad_over_z: &Vec<Array2<f64>>,
    nb_layers: usize,
) -> Vec<Array2<f64>> {
    let mut grad_over_weights = vec![Array2::default((0, 0)); nb_layers];
    for l in (1..nb_layers).rev() {
        grad_over_weights[l] = grad_over_z[l].dot(&values[l - 1].t());
    }
    grad_over_weights
}

fn get_grad_over_biases(grad_over_z: &Vec<Array2<f64>>) -> Vec<Array2<f64>> {
    grad_over_z.clone()
}

fn array2_of_matrix(matrix: &Matrix) -> Array2<f64> {
    Array2::from_shape_fn((matrix.height(), matrix.width()), |(i, j)| matrix.get(i, j))
}

fn matrix_of_array2(array2: &Array2<f64>) -> Matrix {
    let mut matrix = Matrix::new(array2.nrows(), array2.ncols());
    for i in 0..matrix.height() {
        for j in 0..matrix.width() {
            matrix.set(i, j, array2[[i, j]]);
        }
    }
    matrix
}
