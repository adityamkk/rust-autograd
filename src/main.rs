pub mod nn;
use nn::tensor::{Tensor, add, mul, backward};
use std::rc::Rc;
use std::cell::RefCell;
use nalgebra::DMatrix;

fn main() {
    println!("Hello, world!");

    let rows = 3;
    let columns = 3;
    let data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];

    let a = Rc::new(RefCell::new(Tensor::new(DMatrix::<f64>::from_row_slice(rows, columns, &data))));
    let x = Rc::new(RefCell::new(Tensor::new(DMatrix::<f64>::from_row_slice(rows, columns, &data))));
    let b = Rc::new(RefCell::new(Tensor::new(DMatrix::<f64>::from_row_slice(rows, columns, &data))));

    let z_grad = vec![
        0.1, 0.2, -0.1,
        1.1, 5.2, -6.8,
        0.7, 0.4, 2.2,
    ];
    let z = add(&mul(&a, &x), &b);
    backward(&z, DMatrix::<f64>::from_row_slice(rows, columns, &z_grad));

    println!("A Data: {}, A Grad: {}", a.borrow().data, a.borrow().grad);
    println!("X Data: {}, X Grad: {}", x.borrow().data, x.borrow().grad);
    println!("B Data: {}, B Grad: {}", b.borrow().data, b.borrow().grad);
    println!("Z Data: {}, Z Grad: {}", z.borrow().data, z.borrow().grad);
}