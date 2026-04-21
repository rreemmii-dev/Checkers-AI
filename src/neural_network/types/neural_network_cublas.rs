use crate::consts::NeuralNetworkFloat;
use crate::neural_network::neural_network::{NeuralNetworkParameters, NeuralNetworkTrait};
use crate::neural_network::types::cuda::context_pointer::ContextPointer;
use crate::neural_network::types::cuda::cuda::{
    cuda_apply_lhs_rhs_x_times_one_minus_x_then_times, cuda_apply_subtract_with_coef,
    cuda_clone_matrix, cuda_create_context, cuda_create_matrix, cuda_dot, cuda_export_matrix,
    cuda_get, cuda_get_layer_output, cuda_get_result, cuda_import_matrix,
};
use crate::neural_network::types::cuda::matrix_pointer::MatrixPointer;
use crate::neural_network::types::matrix::Matrix;

pub struct NeuralNetworkCuda {
    context: ContextPointer,
    weights: Vec<MatrixPointer>,
    biases: Vec<MatrixPointer>,
    learning_rate: NeuralNetworkFloat,
}

unsafe impl Send for NeuralNetworkCuda {}

impl Clone for NeuralNetworkCuda {
    fn clone(&self) -> Self {
        let context = cuda_create_context();
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
            context,
            weights,
            biases,
            learning_rate,
        }
    }
}

impl NeuralNetworkTrait for NeuralNetworkCuda {
    fn get_output(&self, input: Matrix) -> NeuralNetworkFloat {
        let input = cuda_import_matrix(&input);
        let mut output = input;
        for l in 1..self.weights.len() {
            output =
                cuda_get_layer_output(&self.context, &output, &self.weights[l], &self.biases[l]);
        }
        cuda_get_result(&output, 1)[0]
    }

    fn get_outputs(&self, inputs: Vec<Matrix>) -> Vec<NeuralNetworkFloat> {
        let nb_results = inputs.len();
        let mut matrix = Matrix::new(inputs[0].height(), inputs.len());
        for (column, input) in inputs.into_iter().enumerate() {
            assert_eq!(input.width(), 1);
            for row in 0..input.height() {
                let value = input.get(row, 0);
                matrix.set(row, column, value);
            }
        }
        let input = cuda_import_matrix(&matrix);
        let mut output = input;
        for l in 1..self.weights.len() {
            output =
                cuda_get_layer_output(&self.context, &output, &self.weights[l], &self.biases[l]);
        }
        cuda_get_result(&output, nb_results)
    }

    fn change_learning_rate(&mut self, ratio: NeuralNetworkFloat) {
        self.learning_rate *= ratio;
    }

    fn import(parameters: &NeuralNetworkParameters) -> Self {
        let context = cuda_create_context();
        let weights = parameters
            .weights
            .iter()
            .map(cuda_import_matrix)
            .collect::<Vec<_>>();
        let biases = parameters
            .biases
            .iter()
            .map(cuda_import_matrix)
            .collect::<Vec<_>>();
        let learning_rate = parameters.learning_rate;
        NeuralNetworkCuda {
            context,
            weights,
            biases,
            learning_rate,
        }
    }

    fn export(&self) -> NeuralNetworkParameters {
        let weights = self
            .weights
            .iter()
            .map(cuda_export_matrix)
            .collect::<Vec<_>>();
        let biases = self
            .biases
            .iter()
            .map(cuda_export_matrix)
            .collect::<Vec<_>>();
        let learning_rate = self.learning_rate;
        NeuralNetworkParameters {
            learning_rate,
            weights,
            biases,
        }
    }

    fn train_once(&mut self, input: &Matrix, expected: NeuralNetworkFloat) {
        let nb_layers = self.weights.len();
        let input = cuda_import_matrix(input);
        let values = get_values(&self.context, input, &self.weights, &self.biases, nb_layers);
        let grad_over_z =
            get_grad_over_z(&self.context, &values, expected, &self.weights, nb_layers);
        let grad_over_weights =
            get_grad_over_weights(&self.context, &values, &grad_over_z, nb_layers);
        // let grad_over_biases = get_grad_over_biases(&grad_over_z);
        let grad_over_biases = grad_over_z;
        for l in 1..nb_layers {
            cuda_apply_subtract_with_coef(
                &self.context,
                &mut self.weights[l],
                &grad_over_weights[l],
                self.learning_rate,
            );
            cuda_apply_subtract_with_coef(
                &self.context,
                &mut self.biases[l],
                &grad_over_biases[l],
                self.learning_rate,
            );
        }
    }
}

fn get_values(
    context: &ContextPointer,
    input: MatrixPointer,
    weights: &[MatrixPointer],
    biases: &[MatrixPointer],
    nb_layers: usize,
) -> Vec<MatrixPointer> {
    let mut values = vec![MatrixPointer::null(); nb_layers];
    values[0] = input;
    for l in 1..nb_layers {
        values[l] = cuda_get_layer_output(context, &values[l - 1], &weights[l], &biases[l]);
    }
    values
}

fn get_grad_over_z(
    context: &ContextPointer,
    values: &[MatrixPointer],
    expected: NeuralNetworkFloat,
    weights: &[MatrixPointer],
    nb_layers: usize,
) -> Vec<MatrixPointer> {
    let mut grad_over_z = vec![MatrixPointer::null(); nb_layers];
    for l in (1..nb_layers).rev() {
        let mut grad_over_values = if l == nb_layers - 1 {
            let guessed = cuda_get(&values[values.len() - 1], 0, 0);
            cuda_create_matrix(
                ((1. - expected) / (1. - guessed)) - expected / guessed,
                1,
                1,
            )
        } else {
            cuda_dot(context, &weights[l + 1], true, &grad_over_z[l + 1], false)
        };
        // grad_over_z[l] = grad_over_values * x_times_one_minus_x(values[l])
        cuda_apply_lhs_rhs_x_times_one_minus_x_then_times(
            context,
            &mut grad_over_values,
            &values[l],
        );
        grad_over_z[l] = grad_over_values;
    }
    grad_over_z
}

fn get_grad_over_weights(
    context: &ContextPointer,
    values: &[MatrixPointer],
    grad_over_z: &[MatrixPointer],
    nb_layers: usize,
) -> Vec<MatrixPointer> {
    let mut grad_over_weights = vec![MatrixPointer::null(); nb_layers];
    for l in (1..nb_layers).rev() {
        grad_over_weights[l] = cuda_dot(context, &grad_over_z[l], false, &values[l - 1], true);
    }
    grad_over_weights
}

fn get_grad_over_biases(grad_over_z: &[MatrixPointer]) -> Vec<MatrixPointer> {
    grad_over_z
        .iter()
        .map(cuda_clone_matrix)
        .collect::<Vec<_>>()
}
