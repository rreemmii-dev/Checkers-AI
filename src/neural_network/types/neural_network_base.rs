use crate::consts::NeuralNetworkFloat;
use crate::neural_network::neural_network::{NeuralNetworkParameters, NeuralNetworkTrait};
use crate::neural_network::types::matrix::Matrix;
use ndarray::{Array2, arr2};

#[derive(Clone)]
pub struct NeuralNetworkBase {
    learning_rate: NeuralNetworkFloat,
    weights: Vec<Array2<NeuralNetworkFloat>>,
    biases: Vec<Array2<NeuralNetworkFloat>>,
}

impl NeuralNetworkTrait for NeuralNetworkBase {
    fn get_output(&self, input: Matrix) -> NeuralNetworkFloat {
        let mut output = array2_of_matrix(&input);
        for l in 1..self.weights.len() {
            output = get_layer_output(&output, &self.weights[l], &self.biases[l]);
        }
        output[[0, 0]]
    }

    fn get_outputs(&self, inputs: Vec<Matrix>) -> Vec<NeuralNetworkFloat> {
        let mut output = Array2::from_shape_fn((inputs[0].height(), inputs.len()), |(i, j)| {
            inputs[j].get(i, 0)
        });
        for l in 1..self.weights.len() {
            output = get_layer_output(&output, &self.weights[l], &self.biases[l]);
        }
        let mut res = Vec::new();
        for i in 0..inputs.len() {
            res.push(output[[0, i]]);
        }
        res
    }

    fn change_learning_rate(&mut self, ratio: NeuralNetworkFloat) {
        self.learning_rate *= ratio;
    }

    fn import(parameters: &NeuralNetworkParameters) -> Self {
        let weights = parameters
            .weights
            .iter()
            .map(array2_of_matrix)
            .collect::<Vec<_>>();
        let biases = parameters
            .biases
            .iter()
            .map(array2_of_matrix)
            .collect::<Vec<_>>();
        let learning_rate = parameters.learning_rate;
        Self {
            learning_rate,
            weights,
            biases,
        }
    }

    fn export(&self) -> NeuralNetworkParameters {
        let weights = self
            .weights
            .iter()
            .map(matrix_of_array2)
            .collect::<Vec<_>>();
        let biases = self.biases.iter().map(matrix_of_array2).collect::<Vec<_>>();
        let learning_rate = self.learning_rate;
        NeuralNetworkParameters {
            learning_rate,
            weights,
            biases,
        }
    }

    fn train_once(&mut self, input: &Matrix, expected: NeuralNetworkFloat) {
        let nb_layers = self.weights.len();
        let values = get_values(
            array2_of_matrix(input),
            &self.weights,
            &self.biases,
            nb_layers,
        );
        let grad_over_z = get_grad_over_z(&values, expected, &self.weights, nb_layers);
        let grad_over_weights = get_grad_over_weights(&values, &grad_over_z, nb_layers);
        let grad_over_biases = get_grad_over_biases(&grad_over_z);
        for l in 1..nb_layers {
            self.weights[l] = &self.weights[l] - self.learning_rate * &grad_over_weights[l];
            self.biases[l] = &self.biases[l] - self.learning_rate * &grad_over_biases[l];
        }
    }
}

fn get_layer_output(
    input: &Array2<NeuralNetworkFloat>,
    weight: &Array2<NeuralNetworkFloat>,
    bias: &Array2<NeuralNetworkFloat>,
) -> Array2<NeuralNetworkFloat> {
    sigmoid(&(weight.dot(input) + bias))
}

fn get_values(
    input: Array2<NeuralNetworkFloat>,
    weights: &[Array2<NeuralNetworkFloat>],
    biases: &[Array2<NeuralNetworkFloat>],
    nb_layers: usize,
) -> Vec<Array2<NeuralNetworkFloat>> {
    let mut values = vec![Array2::default((0, 0)); nb_layers];
    values[0] = input;
    for l in 1..nb_layers {
        values[l] = get_layer_output(&values[l - 1], &weights[l], &biases[l]);
    }
    values
}

fn get_grad_over_z(
    values: &[Array2<NeuralNetworkFloat>],
    expected: NeuralNetworkFloat,
    weights: &[Array2<NeuralNetworkFloat>],
    nb_layers: usize,
) -> Vec<Array2<NeuralNetworkFloat>> {
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
    values: &[Array2<NeuralNetworkFloat>],
    grad_over_z: &[Array2<NeuralNetworkFloat>],
    nb_layers: usize,
) -> Vec<Array2<NeuralNetworkFloat>> {
    let mut grad_over_weights = vec![Array2::default((0, 0)); nb_layers];
    for l in (1..nb_layers).rev() {
        grad_over_weights[l] = grad_over_z[l].dot(&values[l - 1].t());
    }
    grad_over_weights
}

fn get_grad_over_biases(
    grad_over_z: &[Array2<NeuralNetworkFloat>],
) -> Vec<Array2<NeuralNetworkFloat>> {
    grad_over_z.to_owned()
}

fn sigmoid(z: &Array2<NeuralNetworkFloat>) -> Array2<NeuralNetworkFloat> {
    z.mapv(float_sigmoid)
}

fn float_sigmoid(x: NeuralNetworkFloat) -> NeuralNetworkFloat {
    1. / (1. + NeuralNetworkFloat::exp(-x))
}

fn array2_of_matrix(matrix: &Matrix) -> Array2<NeuralNetworkFloat> {
    Array2::from_shape_fn((matrix.height(), matrix.width()), |(i, j)| matrix.get(i, j))
}

fn matrix_of_array2(array2: &Array2<NeuralNetworkFloat>) -> Matrix {
    let mut matrix = Matrix::new(array2.nrows(), array2.ncols());
    for i in 0..matrix.height() {
        for j in 0..matrix.width() {
            matrix.set(i, j, array2[[i, j]]);
        }
    }
    matrix
}
