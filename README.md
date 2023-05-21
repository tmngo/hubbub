# hubbub

## Building LLVM on Windows

```powershell
git clone https://github.com/llvm/llvm-project -b release/15.x --depth 1 "C:/Users/Tim/AppData/Local/llvmenv/llvm-project-15"

mkdir C:\Users\Tim\AppData\Local\llvmenv\llvm-project-15-build

cd C:\Users\Tim\AppData\Local\llvmenv\llvm-project-15-build

cmake C:\Users\Tim\AppData\Local\llvmenv\llvm-project-15\llvm ^
  -Thost=x64 ^
  -G "Visual Studio 16 2019" ^
  -A x64 ^
  -DCMAKE_INSTALL_PREFIX=C:\Users\Tim\AppData\Local\llvmenv\llvm-15.0.7-x86_64-windows-msvc-release-mt ^
  -DCMAKE_PREFIX_PATH=C:\Users\Tim\AppData\Local\llvmenv\llvm-15.0.7-x86_64-windows-msvc-release-mt ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLVM_USE_CRT_RELEASE=MT

msbuild /m -p:Configuration=Release INSTALL.vcxproj

$env:LLVM_SYS_150_PREFIX="C:\Users\Tim\AppData\Local\llvmenv\llvm-15.0.7-x86_64-windows-msvc-release-mt"
```

<!-- ```sh
git clone https://github.com/llvm/llvm-project -b release/15.x --depth 1 "C:/Users/Tim/AppData/Local/llvmenv/llvm-project-15"
```

```toml
# C:/Users/Tim/AppData/Roaming/llvmenv/entry.toml
[release-15]
path = "C:/Users/Tim/AppData/Local/llvmenv/release-15/llvm"
build_type = "Release"
# AArch64;AMDGPU;ARM;AVR;BPF;Hexagon;Lanai;LoongArch;Mips;MSP430;NVPTX;PowerPC;RISCV;Sparc;SystemZ;VE;WebAssembly;X86;XCore
# target = ["X86"]
```

```sh
llvmenv build-entry release-15
``` -->
