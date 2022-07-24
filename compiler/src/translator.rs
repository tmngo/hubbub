use crate::jit::CraneliftModule;
use crate::jit::Generator;
use cranelift::prelude::*;
use cranelift::prelude::{Type, Variable};
use cranelift_module::{DataContext, FuncId, Linkage, Module};
use std::collections::HashMap;
use std::path::Path;

struct FunctionTranslator<'a> {
    int: Type,
    builder: FunctionBuilder<'a>,
    variables: HashMap<String, Variable>,
    var_index: usize,
    module: &'a mut Box<dyn CraneliftModule>,
}

impl<'a> Generator<'a> {
    pub fn compile(mut self, filename: &Path) {
        self.define_add_fn();
        let main_id = self.define_main_fn();

        self.state.module.finalize(main_id, filename);
    }

    //
    //
    //
    //

    fn define_add_fn(&mut self) -> FuncId {
        let int = self.state.module.target_config().pointer_type();
        let params = vec!["a", "b"];
        let ret = "result";

        // Create function signature.
        for _ in &params {
            self.ctx.func.signature.params.push(AbiParam::new(int));
        }
        self.ctx.func.signature.returns.push(AbiParam::new(int));

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let mut variables = HashMap::<String, Variable>::new();
        let mut var_index = 0;

        for (i, param) in params.iter().enumerate() {
            let var = declare_variable(int, &mut builder, &mut variables, &mut var_index, param);
            let val = builder.block_params(entry_block)[i];
            builder.def_var(var, val);
        }

        let return_var = declare_variable(int, &mut builder, &mut variables, &mut var_index, ret);
        let return_val = builder.ins().iconst(int, 0);
        builder.def_var(return_var, return_val);

        let a = builder.use_var(*variables.get("a").unwrap());
        let b = builder.use_var(*variables.get("b").unwrap());
        let return_val = builder.ins().iadd(a, b);
        builder.def_var(return_var, return_val);

        let return_variable = variables.get(ret).unwrap();
        let return_value = builder.use_var(*return_variable);
        builder.ins().return_(&[return_value]);

        builder.finalize();
        let id = self
            .state
            .module
            .declare_function("add", Linkage::Export, &self.ctx.func.signature)
            .unwrap();
        self.state
            .module
            .define_function(id, &mut self.ctx)
            .unwrap();
        self.state.module.clear_context(&mut self.ctx);

        return id;
    }

    fn define_main_fn(&mut self) -> FuncId {
        // Parser results.
        // let params = vec!["a", "b"];
        let params = Vec::<&str>::new();

        let body_vars = vec!["c", "d"];
        let body_vals = vec![3, 14];
        // let mut params = vec!["a", "b"];
        let ret = "result";

        let int = self.state.module.target_config().pointer_type();

        // Create the builder to build a function.
        // let signature = Signature::new(isa::CallConv::Fast);
        // let function = Function::with_name_signature(name: ExternalName, sig: Signature)
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
        // Create the entry block to start emitting code.
        let entry_block = builder.create_block();

        for _ in &params {
            builder.func.signature.params.push(AbiParam::new(int));
        }

        builder.func.signature.returns.push(AbiParam::new(int));

        // Emit parameters.

        builder.append_block_params_for_function_params(entry_block);

        builder.switch_to_block(entry_block);

        builder.seal_block(entry_block);

        // Declare parameters and return variables.
        let variables = HashMap::<String, Variable>::new();

        let mut translator = FunctionTranslator {
            int,
            builder,
            variables,
            var_index: 0,
            module: &mut self.state.module,
        };

        for (i, param) in params.iter().enumerate() {
            let var = translator.declare_variable(param);
            // let var = declare_variable(int, &mut builder, &mut variables, &mut var_index, param);
            let val = translator.builder.block_params(entry_block)[i];
            translator.def_var(var, val);
        }

        for (i, name) in body_vars.iter().enumerate() {
            let var = translator.declare_variable(name);
            // let var = declare_variable(int, &mut builder, &mut variables, &mut var_index, name);
            let val = translator.builder.ins().iconst(int, body_vals[i]);
            translator.def_var(var, val);
        }

        let return_var = translator.declare_variable(ret);
        // let return_var = declare_variable(int, &mut builder, &mut variables, &mut var_index, ret);
        let return_val = translator.builder.ins().iconst(int, 0);
        translator.def_var(return_var, return_val);

        // Emit body.

        // let a = builder.use_var(*variables.get("a").unwrap());
        // let b = builder.use_var(*variables.get("b").unwrap());
        let c = translator.builder.use_var(translator.get_var("c"));
        let d = translator.builder.use_var(translator.get_var("d"));
        let return_val = translator.translate_call("putchar", vec![c, d]);
        // let return_val = translator.builder.ins().iadd(c, d);
        translator.def_var(return_var, return_val);

        // Emit return.

        let return_variable = translator.get_var(ret);
        let return_value = translator.builder.use_var(return_variable);

        translator.builder.ins().return_(&[return_value]);

        translator.builder.finalize();

        // Finalize

        let id = self
            .state
            .module
            .declare_function("main", Linkage::Export, &self.ctx.func.signature)
            .unwrap();

        self.state
            .module
            .define_function(id, &mut self.ctx)
            .unwrap();

        //

        self.state.module.clear_context(&mut self.ctx);

        return id;
    }
}

impl<'a> FunctionTranslator<'a> {
    fn declare_variable(&mut self, name: &str) -> Variable {
        let var = Variable::new(self.var_index);
        if !self.variables.contains_key(name) {
            self.variables.insert(name.into(), var);
            self.builder.declare_var(var, self.int);
            self.var_index += 1;
        }
        return var;
    }

    fn get_var(&self, name: &str) -> Variable {
        return *self.variables.get(name).unwrap();
    }

    fn def_var(&mut self, var: Variable, val: Value) {
        self.builder.def_var(var, val);
    }

    fn translate_call(&mut self, name: &str, args: Vec<Value>) -> Value {
        let mut sig = self.module.make_signature();

        for _ in &args {
            sig.params.push(AbiParam::new(self.int));
        }

        sig.returns.push(AbiParam::new(self.int));

        let callee = self
            .module
            .declare_function(name, Linkage::Import, &sig)
            .unwrap();
        let local_callee = self
            .module
            .declare_func_in_func(callee, &mut self.builder.func);

        let call = self.builder.ins().call(local_callee, &args);
        return self.builder.inst_results(call)[0];
    }
}

fn declare_variable(
    int: types::Type,
    builder: &mut FunctionBuilder,
    variables: &mut HashMap<String, Variable>,
    index: &mut usize,
    name: &str,
) -> Variable {
    let var = Variable::new(*index);
    if !variables.contains_key(name) {
        variables.insert(name.to_string(), var);
        builder.declare_var(var, int);
        *index += 1;
    }
    return var;
}
