use crate::types::{ArrayMatrix, ArrayVector};
use nalgebra::DMatrix;

pub trait KalmanFilterModel {
    fn predict_mat(&self, x: &ArrayVector) -> ArrayMatrix;
    fn predict_cov(&self, x: &ArrayVector) -> ArrayMatrix;

    fn observe_mat(&self, x: &ArrayVector) -> ArrayMatrix;
    fn observe_cov(&self, x: &ArrayVector) -> ArrayMatrix;

    fn predict(&self, x: &ArrayVector, p_inpt: &ArrayMatrix) -> (ArrayVector, ArrayMatrix) {
        let f = self.predict_mat(x);
        let q = self.predict_cov(x);

        let x_pred = &f * x;
        let p_pred = &f * p_inpt * &f.transpose() + &q;

        (x_pred, p_pred)
    }

    fn update(&self, x: &ArrayVector, p: &ArrayMatrix, z: &ArrayVector) -> (ArrayVector, ArrayMatrix) {
        let model_size = x.len();
        let i: ArrayMatrix = DMatrix::identity(model_size, model_size);
        let h = self.observe_mat(x);
        let r = self.observe_cov(x);
        let k = p * &h.transpose() * (&h * p * &h.transpose() + &r).try_inverse().unwrap();
        
        let x_hat = x + &k * (z - &h * x);
        let p_hat = (&i - &k * &h) * p * (&i - &k * &h).transpose() + &k * &r * &k.transpose();
    
        (x_hat, p_hat)
    }
}