use crate::types::ArrayVector;
use poloto::{Plotter, PlotNum};
use std::f64::consts::PI;
use std::fmt::Display;
use std::io::Write;

pub struct Metrics {
    orbit_real: Vec<[f64; 2]>,
    orbit_meas: Vec<[f64; 2]>,
    orbit_pred: Vec<[f64; 2]>,
    orbit_corr: Vec<[f64; 2]>,
    circle_foci: Vec<[f64; 2]>,
    error_meas: Vec<[f64; 2]>,
    error_corr: Vec<[f64; 2]>,
}

#[derive(Clone, Copy)]
pub enum MetricType {
    Real,
    Measurement,
    Prediction,
    Correction,
}

impl Metrics {
    pub fn new() -> Self {
        let orbit_real = vec![];
        let orbit_meas = vec![];
        let orbit_pred = vec![];
        let orbit_corr = vec![];
        let circle_foci = vec![];
        let error_meas = vec![];
        let error_corr = vec![];

        Metrics {
            orbit_real,
            orbit_meas,
            orbit_pred,
            orbit_corr,
            circle_foci,
            error_meas,
            error_corr,
        }
    }

    pub fn update_orbit(
        &mut self,
        metric_type: MetricType,
        x: &ArrayVector,
    ) {
        let orbit = match metric_type {
            MetricType::Real => &mut self.orbit_real,
            MetricType::Measurement => &mut self.orbit_meas,
            MetricType::Prediction => &mut self.orbit_pred,
            MetricType::Correction => &mut self.orbit_corr,
        };
        orbit.push([x[0], x[1]]);
    }

    pub fn update_error(
        &mut self,
        metric_type: MetricType,
        step: usize,
        real: &ArrayVector,
        comp: &ArrayVector,
    ) {
        let error = match metric_type {
            MetricType::Measurement => &mut self.error_meas,
            MetricType::Correction => &mut self.error_corr,
            _ => unreachable!(),
        };
        error.push([step as f64, (real - comp).norm() / real.norm()]);
    }

    pub fn update_foci(
        &mut self,
        x: f64,
        y: f64
    ) {
        let num_steps = 1000;
        let radius = 0.05;
        for step in 0..=num_steps {
            let angle = 2.0 * PI * (step as f64) / (num_steps as f64);
            self.circle_foci.push([x + radius * angle.cos(), y + radius * angle.sin()]);
        }
    }

    pub fn plot_to_svg<S: Display, X: PlotNum, Y: PlotNum>(name: S, plotter: Plotter<X, Y>) {
        let img = poloto::disp(|a| poloto::simple_theme_dark(a, plotter));
        let mut file = std::fs::File::create(format!("./output/{}.svg", &name)).unwrap();
        let _ = write!(file, "{}", img);
    }

    pub fn save_as_svg(&self) {
        let plotter_orbit = poloto::plot("Orbit", "x", "y")
            .line("Real", &self.orbit_real)
            .line_fill("Earth", &self.circle_foci)
            .line("Measurement", &self.orbit_meas)
            // .line("Prediction", &self.orbit_pred)
            .line("Kalman Filter", &self.orbit_corr)
            .xmarker(0.0)
            .ymarker(0.0)
            .move_into();

        Self::plot_to_svg("orbit", plotter_orbit);

        let plotter_error = poloto::plot("Relative Error", "step", "error (%)")
            .line("Measurements", &self.error_meas)
            .line("Kalman Filter", &self.error_corr)
            .move_into();

        Self::plot_to_svg("error", plotter_error);
    }
}