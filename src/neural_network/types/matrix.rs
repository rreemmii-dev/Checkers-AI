use crate::consts::NeuralNetworkFloat;

#[derive(Clone)]
pub struct Matrix {
    matrix: Vec<Vec<NeuralNetworkFloat>>,
}

impl Matrix {
    pub fn new(height: usize, width: usize) -> Self {
        Self::zero(height, width)
    }

    pub fn zero(height: usize, width: usize) -> Self {
        let line = vec![0.; width];
        let mut matrix = Vec::new();
        for _ in 0..height {
            matrix.push(line.clone());
        }
        Self { matrix }
    }

    pub fn get(&self, row: usize, column: usize) -> NeuralNetworkFloat {
        self.matrix[row][column]
    }

    pub fn set(&mut self, row: usize, column: usize, value: NeuralNetworkFloat) {
        self.matrix[row][column] = value;
    }

    pub fn height(&self) -> usize {
        self.matrix.len()
    }

    pub fn width(&self) -> usize {
        if self.height() == 0 {
            return 0;
        }
        self.matrix[0].len()
    }
}
