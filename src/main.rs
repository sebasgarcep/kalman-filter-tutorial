mod filters;
mod metrics;
mod models;
mod simulations;
mod types;

use crate::filters::KalmanFilterModel;
use crate::metrics::{Metrics, MetricType};
use crate::models::ConstantOrbitModel;
use crate::simulations::OrbitSimulation;
use crate::types::ArrayVector;
use nalgebra::{dmatrix, dvector};
use rand::rngs::ThreadRng;
use std::f64::consts::PI;

const NUM_STEPS: usize = 1000;
const POSITION_NOISE: f64 = 0.01;
const ANGLE_NOISE: f64 = 2.0 * PI * 0.01;
const A_REAL: f64 = 2.0;
const B_REAL: f64 = 1.0;

fn main() {
    let mut rng = ThreadRng::default();

    let simulation = OrbitSimulation::new(A_REAL, B_REAL, POSITION_NOISE, ANGLE_NOISE);

    // Estimate parameters
    let e_meas = simulation.measure_foci(&mut rng);
    let model = ConstantOrbitModel::new(POSITION_NOISE, e_meas);

    let mut x_corr: ArrayVector = dvector![
        0.0,
        0.0
    ];

    let mut p_corr = dmatrix![
        POSITION_NOISE, 0.0;
        0.0, ANGLE_NOISE;
    ];

    let mut metrics = Metrics::new();
    metrics.update_foci(simulation.c, 0.0);

    for step in 0..NUM_STEPS {
        let t = (step as f64) / (NUM_STEPS as f64);

        let x_real = simulation.calculate_orbit(t);
        metrics.update_orbit(MetricType::Real, &x_real);

        let x_meas = simulation.measure_state(&x_real, &model.e_meas, &mut rng);
        metrics.update_orbit(MetricType::Measurement, &x_meas);
        metrics.update_error(MetricType::Measurement, step, &x_real, &x_meas);

        if step > 0 {
            let (x_pred, p_pred) = model.predict(&x_corr, &p_corr);
            metrics.update_orbit(MetricType::Prediction, &x_pred);

            let (x_hat, p_hat) = model.update(&x_pred, &p_pred, &x_meas);

            x_corr = x_hat;
            p_corr = p_hat;
        } else {
            x_corr = x_meas;
            metrics.update_orbit(MetricType::Prediction, &x_corr);
        }

        metrics.update_orbit(MetricType::Correction, &x_corr);
        metrics.update_error(MetricType::Correction, step, &x_real, &x_corr);
    }

    metrics.save_as_svg();
}