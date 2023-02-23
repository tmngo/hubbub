set windows-shell := ["powershell.exe", "-c"]

cl FILENAME:
    cd examples/tests; cargo run {{FILENAME}}

jit FILENAME:
    cd examples/tests; cargo run {{FILENAME}} -j

llvm FILENAME:
    cd examples/tests; cargo run {{FILENAME}} -r

test TESTNAME='':
    cargo test {{TESTNAME}}

deps := 'target/debug/deps/'
modules := deps + 'modules'
FR := '-Force -Recurse'

cp_modules:
    mkdir {{modules}} -Force
    rm {{modules}}/* {{FR}}
    cp modules {{deps}} {{FR}}
    mv target/debug/test_shared.dll.lib target/debug/test_shared.lib
    
