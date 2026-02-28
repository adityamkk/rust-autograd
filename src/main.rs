pub mod nn;
use nn::tensor::{Tensor, add, mul, backward};
use std::rc::Rc;
use std::cell::RefCell;

fn main() {
    println!("Hello, world!");

    let a = Rc::new(RefCell::new(Tensor::new(3.0)));
    let x = Rc::new(RefCell::new(Tensor::new(4.0)));
    let b = Rc::new(RefCell::new(Tensor::new(5.0)));

    let z = add(&mul(&a, &x), &b);
    backward(&z);

    println!("A Data: {}, A Grad: {}", a.borrow().data, a.borrow().grad);
    println!("X Data: {}, X Grad: {}", x.borrow().data, x.borrow().grad);
    println!("B Data: {}, B Grad: {}", b.borrow().data, b.borrow().grad);
    println!("Z Data: {}, Z Grad: {}", z.borrow().data, z.borrow().grad);
}