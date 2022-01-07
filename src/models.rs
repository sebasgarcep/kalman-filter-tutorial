use crate::filters::ExtendedKalmanFilterModel;
use crate::types::{ArrayMatrix, ArrayVector};
use nalgebra::{dmatrix, DMatrix};

pub struct ConstantOrbitModel {
    pub position_noise: f64,
    pub e_meas: ArrayVector,
}

impl ConstantOrbitModel {
    pub fn new(
        position_noise: f64,
        e_meas: ArrayVector,
    ) -> Self {
        ConstantOrbitModel {
            position_noise,
            e_meas,
        }
    }
}

impl ExtendedKalmanFilterModel for ConstantOrbitModel {
    fn predict_apply(&self, x: &ArrayVector) -> ArrayVector {
        x.clone()
    }

    fn predict_mat(&self, _x: &ArrayVector) -> ArrayMatrix {
        DMatrix::identity(2, 2)
    }

    fn predict_cov(&self, _x: &ArrayVector) -> ArrayMatrix {
        dmatrix![
            self.position_noise, 0.0;
            0.0, self.position_noise;
        ]
    }

    fn observe_apply(&self, x: &ArrayVector) -> ArrayVector {
        x.clone()
    }

    fn observe_mat(&self, _x: &ArrayVector) -> ArrayMatrix {
        DMatrix::identity(2, 2)
    }

    fn observe_cov(&self, x: &ArrayVector) -> ArrayMatrix {
        dmatrix![
            self.position_noise * (x - &self.e_meas).norm_squared(), 0.0;
            0.0, self.position_noise * (x - &self.e_meas).norm_squared();
        ]
    }
}