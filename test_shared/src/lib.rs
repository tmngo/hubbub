#[export_name = "hubbub-test-addi64"]
fn addi64(left: i64, right: i64) -> i64 {
    left + right
}

#[export_name = "hubbub-test-hello"]
fn hello() {
    println!("Hello!");
}

#[export_name = "hubbub-test-one"]
fn one() -> i64 {
    1
}
