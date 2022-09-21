```bash
# Compile to an executable using Cranelift.
cargo run <filename>

# JIT compile and execute using Cranelift.
cargo run <filename> -j

# Compile to an executable using LLVM.
cargo run <filename> -r

# Run all tests
cargo test
cargo test -- --nocapture
```
