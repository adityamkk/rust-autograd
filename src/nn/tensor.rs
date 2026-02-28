use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashSet;
use nalgebra::DMatrix;

type TensorRef = Rc<RefCell<Tensor>>;

pub struct Tensor {
    pub data: DMatrix<f64>,
    pub grad: DMatrix<f64>,
    pub parents: Vec<TensorRef>,
    pub backward: Option<Box<dyn Fn()>>,
}

impl Tensor {
    pub fn new(data: DMatrix<f64>) -> Tensor {

        let nrows = data.nrows();
        let ncols = data.ncols();

        Tensor {
            data: data,
            grad: DMatrix::zeros(nrows, ncols),
            parents: vec![],
            backward: None
        }
    }
}

pub fn add(a: &TensorRef, b: &TensorRef) -> TensorRef {
    // Get dimensions
    let out_rows = a.borrow().data.nrows();
    let out_cols = a.borrow().data.ncols();
    assert_eq!(out_rows, b.borrow().data.nrows());
    assert_eq!(out_cols, b.borrow().data.ncols());

    let out = Rc::new(RefCell::new(Tensor {
        data: &a.borrow().data + &b.borrow().data, // Element-wise addition of matrices
        grad: DMatrix::zeros(out_rows, out_cols),
        parents: vec![a.clone(), b.clone()],
        backward: None
    }));

    // Move cloned strong pointers into the backwards function
    let out_clone = out.clone();
    let a_clone = a.clone();
    let b_clone = b.clone();

    out.borrow_mut().backward = Some(Box::new(move || {
        println!("Get gradient in add");
        let grad = &out_clone.borrow().grad;
        println!("Got gradient in add, add to gradient of first prev...");
        a_clone.borrow_mut().grad += grad;
        println!("Finished first add, add to gradient of second prev...");
        b_clone.borrow_mut().grad += grad;
        println!("Done with add backprop");
    }));

    out
}

pub fn mul(a: &TensorRef, b: &TensorRef) -> TensorRef {
    // Get dimensions

    let a_data = &a.borrow().data;
    let b_data = &b.borrow().data;
    let out_rows = a_data.nrows();
    let out_cols = b_data.ncols();
    assert_eq!(a_data.ncols(), b_data.nrows());

    let out = Rc::new(RefCell::new(Tensor {
        data: &a.borrow().data * &b.borrow().data,
        grad: DMatrix::zeros(out_rows, out_cols),
        parents: vec![a.clone(), b.clone()],
        backward: None
    }));

    // Move cloned strong pointers into the backwards function
    let out_clone = out.clone();
    let a_clone = a.clone();
    let b_clone = b.clone();

    out.borrow_mut().backward = Some(Box::new(move || {
        let grad = &out_clone.borrow().grad;
        let a_data = &mut a_clone.borrow_mut();
        let b_data = &mut b_clone.borrow_mut();
        a_data.grad += grad * b_data.data.transpose(); // dC x B^T
        b_data.grad += a_data.data.transpose() * grad; // A^T x dC (which is also (dC^T x A)^T)
    }));

    out
}

pub fn relu(a: &TensorRef) -> TensorRef {
    // Get dimensions
    let out_rows = a.borrow().data.nrows();
    let out_cols = a.borrow().data.ncols();

    let out = Rc::new(RefCell::new(Tensor {
        data: a.borrow().data.map(|x| {if x > 0.0 { x } else { 0.0 }}),
        grad: DMatrix::zeros(out_rows, out_cols),
        parents: vec![a.clone()],
        backward: None
    }));

    let out_clone = out.clone();
    let a_clone = a.clone();

    out.borrow_mut().backward = Some(Box::new(move || {
        let grad = &out_clone.borrow().grad;
        let a_data = &mut a_clone.borrow_mut();
        let relu_d = a_data.data.map(|x| {if x > 0.0 { 1.0 } else { 0.0 }});
        a_data.grad += grad.component_mul(&relu_d); // Hadamard Product
    }));

    out
}

/**
 * Builds the topological sort of the computational graph
 * Uses DFS
 */
pub fn topological_sort(t: &TensorRef, visited: &mut HashSet<usize>, order: &mut Vec<TensorRef>) {
    let ptr = Rc::as_ptr(t) as usize; // This is just to get a quick hash of the tensor
    if visited.contains(&ptr) {
        // Already found this somewhere else in the dfs, so I'm already counting it
        return
    }
    visited.insert(ptr);

    // Do dfs on the parents
    for parent in &t.borrow().parents {
        topological_sort(parent, visited, order);
    }

    // Once all parents processed, I can finally add myself to the order
    order.push(t.clone());
}

/**
 * Applies a gradient penalty to the tensor, scaled by a learning rate factor
 */
pub fn apply_grad_penalty(t: &TensorRef, lr: f64) {
    let mut t_data = t.borrow_mut();
    let penalty = &t_data.grad * (-1.0 * lr);
    t_data.data += penalty;
}

pub fn reset_grad(t: &TensorRef) {
    let mut t_data = t.borrow_mut();
    let t_rows = t_data.grad.nrows();
    let t_cols = t_data.grad.ncols();

    t_data.grad = DMatrix::zeros(t_rows, t_cols);
}

/**
 * Given a Tensor to compute gradients on, compute gradients on all previous tensors
 * Updates Gradients of Tensors in reverse topological order
 * Gradients are summed up before being used to go backwards
 */
pub fn backward(t: &TensorRef, init_grad: DMatrix<f64>) {
    t.borrow_mut().grad = init_grad; // Initial gradient, eventually change this to some kind of loss function
    
    let mut order = Vec::<TensorRef>::new();
    let mut visited = HashSet::<usize>::new();

    topological_sort(t, &mut visited, &mut order);

    // Iterate topological sort in reverse order to compute gradients
    for tensor in order.into_iter().rev() {
        if let Some(ref backward_fn) = tensor.borrow().backward {
            backward_fn(); // Use current gradient to update gradients of parents
        }
        apply_grad_penalty(&tensor, 0.2);
        reset_grad(&tensor);
    }
}

pub fn loss_mse(y_obs: &DMatrix<f64>, y_exp: &DMatrix<f64>) -> f64 {
    assert_eq!(y_obs.shape(), y_exp.shape());

    let mut sum = 0.0;

    for (x, y) in y_obs.iter().zip(y_exp.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }

    sum / (y_obs.len() as f64)
}

pub fn loss_mse_grad(y_obs: &DMatrix<f64>, y_exp: &DMatrix<f64>) -> DMatrix<f64> {
    assert_eq!(y_obs.shape(), y_exp.shape());

    (y_obs - y_exp) * 2.0
}