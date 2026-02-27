extern crate nalgebra as na;
use na::{DMatrix};

pub struct Layer {
    in_dim: usize,
    out_dim: usize,
    activation_fn_str: String,
    weights: DMatrix<f64>,
    activation_fn: fn(DMatrix<f64>) -> DMatrix<f64>,
    der_activation_fn: fn(DMatrix<f64>) -> DMatrix<f64>
}

impl Layer {

    pub fn new(in_dim: usize, out_dim: usize, af: String) -> Layer {
        // TODO: Write activation functions and derivatives of activation functions
        Layer {
            in_dim: in_dim,
            out_dim: out_dim,
            activation_fn_str: af,
            weights: DMatrix::<f64>::zeros(in_dim, out_dim)
        }
    }

    pub fn shape(self) -> (usize, usize) {
        (self.in_dim, self.out_dim)
    }

    pub fn get_af(&self) -> &String {
        &self.activation_fn_str
    }
}