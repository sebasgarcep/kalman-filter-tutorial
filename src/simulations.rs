use crate::types::ArrayVector;
use nalgebra::dvector;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

pub struct OrbitSimulation {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub e: ArrayVector,
    eccentricity: f64,
    position_noise_distribution: Normal<f64>,
    angle_noise_distribution: Normal<f64>,
}

impl OrbitSimulation {
    pub fn new(a: f64, b: f64, position_noise: f64, angle_noise: f64) -> Self {
        let c = (a.powi(2) - b.powi(2)).sqrt();
        let e = dvector![
            c,
            0.0
        ];

        let eccentricity = c / a;

        let position_noise_distribution = Normal::new(0.0, position_noise).unwrap();
        let angle_noise_distribution = Normal::new(0.0, angle_noise).unwrap();

        OrbitSimulation {
            a,
            b,
            c,
            e,
            eccentricity,
            position_noise_distribution,
            angle_noise_distribution,
        }
    }

    fn calculate_kepler_formula_newton(
        &self,
        t: f64,
    ) -> f64 {
        let mean_anomaly = 2.0 * PI * t;
        let mut eccentric_anomaly = if self.eccentricity > 0.8 { PI } else { mean_anomaly };
    
        while (eccentric_anomaly - self.eccentricity * eccentric_anomaly.sin() - mean_anomaly).abs() > 1.0e-8 {
            let numerator = eccentric_anomaly - self.eccentricity * eccentric_anomaly.sin() - mean_anomaly;
            let denominator = 1.0 - self.eccentricity * eccentric_anomaly.cos();
            eccentric_anomaly = eccentric_anomaly - numerator / denominator;
        }
    
        eccentric_anomaly
    }
    
    pub fn calculate_orbit(
        &self,
        t: f64,
    ) -> ArrayVector {
        let eccentric_anomaly = self.calculate_kepler_formula_newton(t);
        let x = self.a * eccentric_anomaly.cos();
        let y = self.b * eccentric_anomaly.sin();
    
        dvector![x, y]
    }

    fn cartesian_to_polar(x: &ArrayVector) -> ArrayVector {
        dvector![
            x.norm(),
            x[1].atan2(x[0])
        ]
    }
    
    fn polar_to_cartesian(x: &ArrayVector) -> ArrayVector {
        dvector![
            x[0] * x[1].cos(),
            x[0] * x[1].sin()
        ]
    }

    fn measure_periapsis<R: Rng>(&self, rng: &mut R) -> f64 {
        self.a - self.c + self.position_noise_distribution.sample(rng)
    }

    fn measure_apoapsis<R: Rng>(&self, rng: &mut R) -> f64 {
        self.a + self.c + self.position_noise_distribution.sample(rng)
    }

    pub fn measure_foci<R: Rng>(&self, rng: &mut R) -> ArrayVector {
        let periapsis_meas = self.measure_periapsis(rng);
        let apoapsis_meas = self.measure_apoapsis(rng);
        let c_meas = (apoapsis_meas - periapsis_meas) / 2.0;
        
        dvector![c_meas, 0.0]
    }

    pub fn measure_state<R: Rng>(&self, x_real: &ArrayVector, e_meas: &ArrayVector, rng: &mut R) -> ArrayVector {
        let z_real = Self::cartesian_to_polar(&(x_real - &self.e));

        let z_meas = dvector![
            z_real[0] + self.position_noise_distribution.sample(rng),
            z_real[1] + self.angle_noise_distribution.sample(rng)
        ];

        Self::polar_to_cartesian(&z_meas) + e_meas
    }
}