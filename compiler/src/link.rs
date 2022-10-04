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

pub fn link(object_filename: &str, output_filename: &str, base_dir: &str) {
    let host = &format!("{}", Triple::host());
    let tool = cc::Build::new()
        .host(host)
        .target(host)
        .opt_level(0)
        .get_compiler();

    let mut command = tool.to_command();

    let lib_path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join(base_dir)
        .join("hubbub_runtime.lib");

    // Static libs are found via absolute path or in PATH.
    // Dynamic libs are found next to the executable or in PATH.
    if tool.is_like_msvc() {
        command.args([
            &format!("-Fe{}", output_filename),
            object_filename,
            lib_path.to_str().unwrap(),
            "advapi32.lib",
            "bcrypt.lib",
            "kernel32.lib",
            "userenv.lib",
            "ws2_32.lib",
            "msvcrt.lib",
            "C:\\Users\\Tim\\Projects\\hubbub\\target\\debug\\glfw3dll.lib",
            "C:\\Users\\Tim\\Projects\\hubbub\\target\\debug\\SimpleDLL.lib",
        ]);
    } else if tool.is_like_clang() {
        command.args([
            &format!("-o{}", output_filename),
            object_filename,
            lib_path.to_str().unwrap(),
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
    assert!(output.status.success());
}

// #[cfg(not(windows))]
pub fn link_gcc(object_filename: &str, output_filename: &str) {
    let mut command = Command::new("gcc");
    command.args([
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
    assert!(output.status.success());
}
