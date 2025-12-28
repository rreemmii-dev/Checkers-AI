#[derive(Clone)]
pub struct Matrix {
    matrix: Vec<Vec<f64>>,
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

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.matrix[i][j]
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.matrix[i][j] = value
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
