use crate::NB_LAYERS;
use crate::players::neural_network::neural_networks_types::cuda::cuda::{
    cuda_clone_matrix, cuda_create_handle, cuda_create_matrix, cuda_dot, cuda_get,
    cuda_import_matrix, cuda_plus, cuda_sigmoid, cuda_subtract_with_coef, cuda_times,
    cuda_x_times_one_minus_x,
};
use crate::players::neural_network::neural_networks_types::cuda::handle_pointer::HandlePointer;
use crate::players::neural_network::neural_networks_types::cuda::matrix_pointer::MatrixPointer;
use crate::players::neural_network::neural_networks_types::matrix::Matrix;
use crate::players::neural_network::neural_networks_types::neural_network::NeuralNetworkTrait;

// TODO: Profiling

pub struct NeuralNetworkCuda {
    handle: HandlePointer,
    weights: Vec<MatrixPointer>,
    biases: Vec<MatrixPointer>,
    learning_rate: f64,
}

unsafe impl Sync for NeuralNetworkCuda {}

unsafe impl Send for NeuralNetworkCuda {}

impl Clone for NeuralNetworkCuda {
    fn clone(&self) -> Self {
        let handle = cuda_create_handle();
        let weights = self
            .weights
            .iter()
            .map(cuda_clone_matrix)
            .collect::<Vec<_>>();
        let biases = self
            .biases
            .iter()
            .map(cuda_clone_matrix)
            .collect::<Vec<_>>();
        let learning_rate = self.learning_rate;
        NeuralNetworkCuda {
            handle,
            weights,
            biases,
            learning_rate,
        }
    }
}

impl NeuralNetworkTrait for NeuralNetworkCuda {
    fn get_output(&self, input: Matrix) -> f64 {
        let input = cuda_import_matrix(&input);
        let mut output = input;
        for l in 1..NB_LAYERS {
            output = get_layer_output(&self.handle, &output, &self.weights[l], &self.biases[l]);
        }
        cuda_get(&output, 0, 0)
    }

    fn get_outputs(&self, inputs: Vec<Matrix>) -> Vec<f64> {
        inputs
            .into_iter()
            .map(|i| self.get_output(i))
            .collect::<Vec<_>>()
        // TODO: Gives error...
        // let input = Array2::from_shape_fn((NODES_PER_LAYER[0], inputs.len()), |(i, j)| {
        //     inputs[j][[i, 0]]
        // });
        // let mut output = cuda_import_matrix(&input);
        // for l in 1..NB_LAYERS {
        //     output = get_layer_output(self.handle, output, self.weights[l], self.biases[l]);
        // }
        // let mut res = Vec::new();
        // for i in 0..inputs.len() {
        //     res.push(cuda_get(output, 0, i));
        // }
        // res
    }

    fn change_learning_rate(&mut self, ratio: f64) {
        self.learning_rate *= ratio;
    }

    fn import(parameters: (Vec<Matrix>, Vec<Matrix>, f64)) -> Self {
        let handle = cuda_create_handle();
        let (weights, biases, learning_rate) = parameters;
        let weights = weights.iter().map(cuda_import_matrix).collect::<Vec<_>>();
        let biases = biases.iter().map(cuda_import_matrix).collect::<Vec<_>>();
        NeuralNetworkCuda {
            handle,
            weights,
            biases,
            learning_rate,
        }
    }

    fn export(&self) -> (Vec<Matrix>, Vec<Matrix>, f64) {
        todo!()
    }

    fn train_once(&mut self, input: Matrix, expected: f64) {
        let input = cuda_import_matrix(&input);
        let values = get_values(&self.handle, &input, &self.weights, &self.biases, NB_LAYERS);
        let grad_over_z =
            get_grad_over_z(&self.handle, &values, expected, &self.weights, NB_LAYERS);
        let grad_over_weights =
            get_grad_over_weights(&self.handle, &values, &grad_over_z, NB_LAYERS);
        let grad_over_biases = get_grad_over_biases(&grad_over_z);
        for l in 1..NB_LAYERS {
            self.weights[l] = cuda_subtract_with_coef(
                &self.handle,
                &self.weights[l],
                &grad_over_weights[l],
                self.learning_rate,
            );
            self.biases[l] = cuda_subtract_with_coef(
                &self.handle,
                &self.biases[l],
                &grad_over_biases[l],
                self.learning_rate,
            );
        }
    }
}

fn get_layer_output(
    handle: &HandlePointer,
    input: &MatrixPointer,
    weight: &MatrixPointer,
    bias: &MatrixPointer,
) -> MatrixPointer {
    cuda_sigmoid(&cuda_plus(
        handle,
        &cuda_dot(handle, weight, false, input, false),
        &bias,
    ))
}

fn get_values(
    handle: &HandlePointer,
    input: &MatrixPointer,
    weights: &Vec<MatrixPointer>,
    biases: &Vec<MatrixPointer>,
    nb_layers: usize,
) -> Vec<MatrixPointer> {
    let mut values = vec![MatrixPointer::null(); nb_layers];
    values[0] = cuda_clone_matrix(input);
    for l in 1..nb_layers {
        values[l] = get_layer_output(handle, &values[l - 1], &weights[l], &biases[l]);
    }
    values
}

fn get_grad_over_z(
    handle: &HandlePointer,
    values: &Vec<MatrixPointer>,
    expected: f64,
    weights: &Vec<MatrixPointer>,
    nb_layers: usize,
) -> Vec<MatrixPointer> {
    let mut grad_over_z = vec![MatrixPointer::null(); nb_layers];
    for l in (1..nb_layers).rev() {
        let grad_over_values = if l == nb_layers - 1 {
            let guessed = cuda_get(&values[values.len() - 1], 0, 0);
            cuda_create_matrix(
                1,
                1,
                ((1. - expected) / (1. - guessed)) - expected / guessed,
            )
        } else {
            cuda_dot(handle, &weights[l + 1], true, &grad_over_z[l + 1], false)
        };
        grad_over_z[l] = cuda_times(&grad_over_values, cuda_x_times_one_minus_x(&values[l]));
    }
    grad_over_z
}

fn get_grad_over_weights(
    handle: &HandlePointer,
    values: &Vec<MatrixPointer>,
    grad_over_z: &Vec<MatrixPointer>,
    nb_layers: usize,
) -> Vec<MatrixPointer> {
    let mut grad_over_weights = vec![MatrixPointer::null(); nb_layers];
    for l in (1..nb_layers).rev() {
        grad_over_weights[l] = cuda_dot(handle, &grad_over_z[l], false, &values[l - 1], true);
    }
    grad_over_weights
}

fn get_grad_over_biases(grad_over_z: &Vec<MatrixPointer>) -> Vec<MatrixPointer> {
    grad_over_z
        .iter()
        .map(cuda_clone_matrix)
        .collect::<Vec<_>>()
}
