# Basic Autograd Engine for Rust

Given a function on tensors, calculates gradients for all tensors in the function.
Traverses the reverse topological ordering of the computational graph and performs gradient descent.
Uses a stored lambda to apply the correct gradient computation at each tensor.

Currently, the only supported operations are multiply and add 2D matrices.

# Installation
You will need the Rust toolchain to run this code.

Run
```
cargo run
```

to run the program.