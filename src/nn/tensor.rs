use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashSet;

type TensorRef = Rc<RefCell<Tensor>>;

pub struct Tensor {
    pub data: f64,
    pub grad: f64,
    pub parents: Vec<TensorRef>,
    pub backward: Option<Box<dyn Fn()>>,
}

impl Tensor {
    pub fn new(data: f64) -> Tensor {
        Tensor {
            data: data,
            grad: 0.0,
            parents: vec![],
            backward: None
        }
    }
}

pub fn add(a: &TensorRef, b: &TensorRef) -> TensorRef {
    let out = Rc::new(RefCell::new(Tensor {
        data: a.borrow().data + b.borrow().data,
        grad: 0.0,
        parents: vec![a.clone(), b.clone()],
        backward: None
    }));

    // Move cloned strong pointers into the backwards function
    let out_clone = out.clone();
    let a_clone = a.clone();
    let b_clone = b.clone();

    out.borrow_mut().backward = Some(Box::new(move || {
        let grad = out_clone.borrow().grad;
        a_clone.borrow_mut().grad += grad;
        b_clone.borrow_mut().grad += grad;
    }));

    out
}

pub fn mul(a: &TensorRef, b: &TensorRef) -> TensorRef {
    let out = Rc::new(RefCell::new(Tensor {
        data: a.borrow().data * b.borrow().data,
        grad: 0.0,
        parents: vec![a.clone(), b.clone()],
        backward: None
    }));

    // Move cloned strong pointers into the backwards function
    let out_clone = out.clone();
    let a_clone = a.clone();
    let b_clone = b.clone();

    out.borrow_mut().backward = Some(Box::new(move || {
        let grad = out_clone.borrow().grad;
        let a_data = a_clone.borrow().data;
        let b_data = b_clone.borrow().data;
        a_clone.borrow_mut().grad += grad * b_data;
        b_clone.borrow_mut().grad += grad * a_data;
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
 * Given a Tensor to compute gradients on, compute gradients on all previous tensors
 * Updates Gradients of Tensors in reverse topological order
 * Gradients are summed up before being used to go backwards
 */
pub fn backward(t: &TensorRef) {
    t.borrow_mut().grad = 1.0; // Initial gradient, eventually change this to some kind of loss function
    
    let mut order = Vec::<TensorRef>::new();
    let mut visited = HashSet::<usize>::new();

    topological_sort(t, &mut visited, &mut order);

    // Iterate topological sort in reverse order to compute gradients
    for tensor in order.into_iter().rev() {
        if let Some(ref backward_fn) = tensor.borrow().backward {
            backward_fn(); // Use current gradient to update gradients of parents
        }
    }
}