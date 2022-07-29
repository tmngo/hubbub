use std::io::{self, Write};
use std::process::Command;
use target_lexicon::Triple;

/**
 * cl.exe /Fe:main filename.obj /Fe:filename.exe /link /defaultlib:libcmt
 * link.exe filename.obj /out:filename.exe /defaultlib:libcmt
 *
 * clang -o a.exe filename
 * link -out:a.exe
 *
 *
 *
 *
 */

pub fn link(object_filename: &str, output_filename: &str) {
    println!("--- BEGIN LINK [host: {}]", Triple::host());
    let host = &format!("{}", Triple::host());
    let tool = cc::Build::new()
        .host(host)
        .target(host)
        .opt_level(0)
        .get_compiler();

    let mut command = tool.to_command();

    if tool.is_like_msvc() {
        command.args([
            &format!("-Fe{}", output_filename),
            object_filename,
            "target/debug/hubbub_runtime.lib",
            "advapi32.lib",
            "bcrypt.lib",
            "kernel32.lib",
            "userenv.lib",
            "ws2_32.lib",
            "msvcrt.lib",
        ]);
    } else if tool.is_like_clang() {
        command.args(&[
            &format!("-o{}", output_filename),
            object_filename,
            "target/debug/hubbub_runtime.lib",
            "-ladvapi32",
            "-lbcrypt",
            "-lkernel32",
            "-luserenv",
            "-lws2_32",
            "-lmsvcrt",
            "-nostdlib",
        ]);
    } else {
        println!("Could not find clang or cl.");
        return;
    };

    println!("Command: {:?}", command);

    let output = command.output().expect("failed to link");
    println!("Status:  {}", output.status);
    io::stdout().write_all(&output.stdout).unwrap();
    io::stderr().write_all(&output.stderr).unwrap();
    println!("--- END LINK\n");
    assert!(output.status.success());
}

// #[cfg(not(windows))]
pub fn link_gcc(object_filename: &str, output_filename: &str) {
    println!("--- BEGIN LINK [host: {}]", Triple::host());

    let mut command = Command::new("gcc");
    command.args(&[
        &format!("-o{}", output_filename),
        object_filename,
        "target/debug/libhubbub_runtime.a",
        "-ldl",
        "-pthread",
    ]);

    println!("Command: {:?}", command);

    let output = command.output().expect("failed to link");
    println!("Status:  {}", output.status);
    io::stdout().write_all(&output.stdout).unwrap();
    io::stderr().write_all(&output.stderr).unwrap();
    println!("--- END LINK\n");
    assert!(output.status.success());
}
