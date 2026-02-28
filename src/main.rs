pub mod nn;
use nn::tensor::{Tensor, add, mul, backward};
use std::rc::Rc;
use std::cell::RefCell;
use nalgebra::DMatrix;

fn main() {
    println!("Hello, world!");

    let rows = 2;
    let columns = 2;
    let data = vec![
        1.0, 2.0,
        4.0, 5.0,
    ];

    let a = Rc::new(RefCell::new(Tensor::new(DMatrix::<f64>::from_row_slice(rows, columns, &data))));
    let x = Rc::new(RefCell::new(Tensor::new(DMatrix::<f64>::from_row_slice(rows, columns, &data))));
    let b = Rc::new(RefCell::new(Tensor::new(DMatrix::<f64>::from_row_slice(rows, columns, &data))));

    let z_grad = vec![
        0.1, 0.2,
        1.1, 5.2,
    ];
    let z = add(&mul(&a, &x), &b);
    backward(&z, DMatrix::<f64>::from_row_slice(rows, columns, &z_grad));

    println!("A Data: {}", a.borrow().data);
    println!("X Data: {}", x.borrow().data);
    println!("B Data: {}", b.borrow().data);
    println!("Z Data: {}", z.borrow().data);
}