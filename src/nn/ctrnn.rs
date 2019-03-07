use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

/// The numeric type that will be used in the neural network
pub type Num = f32;

#[allow(missing_docs)]
#[derive(Debug)]
pub struct Ctrnn<'a> {
    pub y: &'a [Num],
    pub delta_t: Num,
    pub tau: &'a [Num], //time constant
    pub wij: &'a [Num], //weights
    pub theta: &'a [Num], //bias
    pub i: &'a [Num], //sensors
}


#[allow(missing_docs)]
impl<'a> Ctrnn<'a> {
    pub fn activate_nn(&self, steps: usize) -> Vec<Num> {
        let mut y = Ctrnn::vector_to_column_matrix(self.y);
        let theta = Ctrnn::vector_to_column_matrix(self.theta);
        let wij = Ctrnn::vector_to_matrix(self.wij);
        let i = Ctrnn::vector_to_column_matrix(self.i);
        let tau = Ctrnn::vector_to_column_matrix(self.tau);
        let delta_t_tau = tau.apply(&(|x| 1.0 / x)) * self.delta_t;

        for _ in 0..steps {
            let activations = (&y + &theta).apply(&Ctrnn::sigmoid);
            y = &y + delta_t_tau.elemul(
                &((&wij * activations) - &y + &i)
            );
        };
        y.into_vec()
    }

    fn sigmoid(y: Num) -> Num {
        // if y > 0.0 {
            // 1.0
        // } else {
            // 0.0
        // }
        1.0 / (1.0 + (-y).exp())
    }

    fn vector_to_column_matrix(vector: &[Num]) -> Matrix<Num> {
        Matrix::new(vector.len(), 1, vector)
    }

    fn vector_to_matrix(vector: &[Num]) -> Matrix<Num> {
        let width = (vector.len() as Num).sqrt() as usize;
        Matrix::new(width, width, vector)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    macro_rules! assert_delta_vector {
        ($x:expr, $y:expr, $d:expr) => {
            for pos in 0..$x.len() {
                if !(($x[pos] - $y[pos]).abs() <= $d) {
                    panic!(
                        "Element at position {:?} -> {:?} \
                         is not equal to {:?}",
                        pos, $x[pos], $y[pos]
                    );
                }
            }
        };
    }

    #[test]
    fn neural_network_activation_stability() {
        // TODO
        // This test should just ensure that a stable neural network implementation doesn't change
    }
}
