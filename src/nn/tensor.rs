use std::cell::RefCell;
use std::rc::Rc;

type TensorRef = Rc<RefCell<Tensor>>;

pub struct Tensor {
    pub data: f64,
    pub grad: f64,
    pub parents: Vec<TensorRef>,
    pub backward: Option<Box<dyn Fn()>>,
}