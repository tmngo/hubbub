```bash
cargo test
cargo test -- --nocapture
```

```bash
lld-link /entry:main /subsystem:windows /out:executable.exe object.o
gcc -o executable object.o
cl structs.obj ./target/debug/hubbub_runtime.lib

cl /Fe:structs.exe structs.obj target/debug/hubbub_runtime.lib advapi32.lib bcrypt.lib kernel32.lib userenv.lib ws2_32.lib msvcrt.lib

clang -ostructs.exe structs.obj target/debug/hubbub_runtime.lib -lmsvcrt -nostdlib -ladvapi32 -lbcrypt -lkernel32 -luserenv -lws2_32 -lmsvcrt -nostdlib
```

https://github.com/bytecodealliance/cranelift-jit-demo

https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/docs/ir.md

https://github.com/jfecher/ante/tree/master/src/cranelift_backend

https://github.com/amethyst-lang/amethyst/blob/main/src/backend/mod.rs

```
lz4 :: #foreign_library "liblz4";

LZ4_compressBound :: (inputSize: s32) -> s32 #foreign lz4;
LZ4_compress_fast :: (source: *u8, dest: *u8, sourceSize: s32, maxDestSize: s32, acceleration: s32) -> s32 #foreign lz4;
LZ4_sizeofState :: () -> s32 #foreign lz4;
```

``` sh
$ RUSTFLAGS="--print=native-static-libs" cargo build
```

```
      Size    Addr  Type              
C++   malloc  free  new     delete
Odin  alloc   free  new     delete
Jai   alloc   free  New 
Zig   alloc   free  create  destroy
```

### Implicit context

```rust
Allocator :: (
  mode: AllocatorMode, 
  size: i64, 
  allocator_data: Pointer{void}, 
  // For reallocation
  old_size: i64, 
  old_memory_pointer: Pointer{void}, 
  // Only for special cases.
  options: s64
) -> void;

AllocatorMode :: enum
  ALLOCATE
  RESIZE
  FREE
  FREE_ALL
end

Logger :: (
  message: string, 
  ident: string, 
  mode: LogMode, 
  data: Pointer{void}
) -> void;

LogMode :: enum
  NONE
  MINIMAL
  EVERYDAY
  VERBOSE
end

Context :: struct
  thread_index:   u32
  user_index:     u32
  user_data:      Pointer{void}
  allocator:      Allocator
  allocator_data: Pointer{void}
  logger:         Logger
  logger_data:    Pointer{void}
end
```
Demo: Implicit Context
14:04
You don't want to have to manually pass allocators everywhere and you don't want to put them in global variables because you have this problem where

Let's say I've got some really cool library like a regular expression library that will search through strings and do all these string operations and that regular expression library wants to allocate and in a language like C or C++ or even you know D or something it's going to end up using the global allocator right unless the library provides you an API for setting the allocator. 

But often people don't do that, and sometimes you may even want separate allocators for seperate invocations of the library or what have you.

So how do we do that and how do we let you get good code reuse under different memory conditions for code where someone wasn't thinking about allocation like, "can we let people write code where they're not thinking that
hard about how you want the allocation to happen and still let you use it under different allocators, right?" and the
answer is yes and so what we do is we supply in the language an implicit
context, sort of like a `this` pointer in C++ where it invisibly gets passed to every function that's a member function, except here it's a thing that invisibly gets passed to every function, period, and it contains a number of things that you may want to overload. 

You can sort of think of it as inheriting global utility functions. You don't inherit it in a class-hierarchy kind of way, you inherit it in a call-graph tree kind of way.

a) Store allocator in global variable.
 - Causes thread safety problems.

b) Explicitly pass allocator.
- function coloring
- It's inconvenient to thread parameters through every function.

c) Store allocator in thread local variable.
- doesn't work with CTFE as touching globals is not allowed

d) Implicitly pass allocator.
- C ABI problems.
- You're passing around an extra argument to every function. 
Isn't that slow? No, not really, compared to the kinds of things we're optimizing for, which is memory accesses. If you set up the calling convention knowing this is going to happen, you can sort of reserve a register for this, so that if you don't change the context, and you don't call into a function that has extreme register pressure, then it's literally zero cost because the register is unchanged.


### Function overloading

```
[5:34 AM] bakk [kalker, elk]: How would you store functions with the same name but different parameters in a symbol table? Like, function overloading
[5:35 AM] bakk [kalker, elk]: and how would that be handled in code-gen (LLVM) :thonkturtle:
[5:37 AM] MrSmith33: usually overload set is stored which then contains all functions in the overload set
[5:37 AM] MrSmith33: @bakk [kalker, elk]
[5:37 AM] bakk [kalker, elk]: oh
[5:38 AM] bakk [kalker, elk]: in what way :thinkong:
[5:39 AM] MrSmith33: when you register functions in a scope and you have a second function wrap them in a new overload set. If name already maps to overload set add new function to it. Then when resolving the name you can choose one from the overload set
```


```c
int foo (int a, int b) {
  return a + b;
}

int main (void) {
  foo(3, 5);
  return 0;
}
```