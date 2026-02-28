pub mod nn;
use nn::tensor::{Tensor, add, mul, relu, backward, loss_mse, loss_mse_grad};
use std::rc::Rc;
use std::cell::RefCell;
use nalgebra::DMatrix;

fn main() {
    println!("Hello, world!");

    // Learn XOR

    // NN Settings
    let weights1 = vec![
        0.8, -0.5,
        -0.7, 0.5
    ];
    let biases1 = vec![
        0.2, 0.4, -0.3, 0.1, -0.2, 0.2,
        0.3, -0.6, 0.2, 0.2, 0.3, 0.4
    ];
    let weights2 = vec![
        0.3, 0.5,
    ];
    let biases2 = vec![
        0.2, 0.4, -0.3, 0.1, -0.2, 0.2,
    ];

    // Input
    let data = vec![
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
    ];
    let labels = vec![
        0.0, 1.0, 0.0, 1.0, 1.0, 0.0
    ];

    let w1 = Rc::new(RefCell::new(Tensor::new(DMatrix::<f64>::from_row_slice(2, 2, &weights1))));
    let b1 = Rc::new(RefCell::new(Tensor::new(DMatrix::<f64>::from_row_slice(2, 6, &biases1))));

    let w2 = Rc::new(RefCell::new(Tensor::new(DMatrix::<f64>::from_row_slice(1, 2, &weights2))));
    let b2 = Rc::new(RefCell::new(Tensor::new(DMatrix::<f64>::from_row_slice(1, 6, &biases2))));

    for _i in 1..30 {
        let x = Rc::new(RefCell::new(Tensor::new(DMatrix::<f64>::from_row_slice(2, 6, &data))));
        let y = relu(&add(&mul(&w2, &relu(&add(&mul(&w1, &x), &b1))), &b2));
        let y_grad = loss_mse_grad(&y.borrow().data, &DMatrix::<f64>::from_row_slice(1, 6, &labels));
        backward(&y, y_grad);

        // println!("Weights: {}", w.borrow().data);
        //println!("X Data: {}", x.borrow().data);
        //println!("B Data: {}", b.borrow().data);
        let y_data = &y.borrow().data;
        println!("Y OBS: {}", &y_data);
        println!("LOSS MSE: {}", loss_mse(&y_data, &DMatrix::<f64>::from_row_slice(1, 6, &labels)))
    }
}