#[export_name = "hubbub-test-add"]
fn add(left: i64, right: i64) -> i64 {
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
