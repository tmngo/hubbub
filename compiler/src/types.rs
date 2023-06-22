use std::rc::Rc;

pub type TypeId = usize;
pub type TypeIds = Rc<[TypeId]>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    None,
    Never,
    Void,
    Any,
    Boolean,
    // IntegerLiteral,
    // String,
    Type {
        ty: TypeId,
    },
    Array {
        typ: TypeId,
        length: usize,
        is_generic: bool,
    },
    Function {
        parameters: TypeIds,
        returns: Vec<TypeId>,
    },
    Pointer {
        typ: TypeId,
        is_generic: bool,
    },
    Struct {
        fields: Vec<TypeId>,
        is_generic: bool,
    },
    Tuple {
        fields: Vec<TypeId>,
    },
    Parameter {
        index: usize,
        binding: TypeId,
    },
    TypeParameter {
        index: usize,
        binding: TypeId,
    },
    Numeric {
        literal: bool,
        floating: bool,
        signed: bool,
        bytes: u8,
    },
    NumericLiteral {
        floating: bool,
        signed: bool,
        min: TypeId,
        max: TypeId,
    },
}

pub fn integer_type(literal: bool, signed: bool, bytes: u8) -> Type {
    Type::Numeric {
        literal,
        floating: false,
        signed,
        bytes,
    }
}

pub fn integer_literal_type(x: i64) -> Type {
    let min = if (-128..=127).contains(&x) {
        T::I8
    } else if (-32768..=32767).contains(&x) {
        T::I16
    } else if (-2147483648..=2147483647).contains(&x) {
        T::I32
    } else {
        T::I64
    } as TypeId;
    Type::NumericLiteral {
        floating: false,
        signed: true,
        min,
        max: T::I64 as TypeId,
    }
}

pub fn float_type(literal: bool, bytes: u8) -> Type {
    Type::Numeric {
        literal,
        floating: true,
        signed: true,
        bytes,
    }
}
impl Type {
    pub fn parameters(&self) -> &TypeIds {
        if let Type::Function { parameters, .. } = self {
            parameters
        } else {
            unreachable!()
        }
    }
    pub fn returns(&self) -> &Vec<TypeId> {
        if let Type::Function { returns, .. } = self {
            returns
        } else {
            unreachable!()
        }
    }
    pub fn is_signed(&self) -> bool {
        match self {
            Self::Numeric { signed, .. } => *signed,
            _ => unreachable!("is_signed is only valid for integer types: got {:?}", self),
        }
    }
    pub fn min_max_mut(&mut self) -> (&mut TypeId, &mut TypeId) {
        if let Type::NumericLiteral {
            ref mut min,
            ref mut max,
            ..
        } = self
        {
            (min, max)
        } else {
            unreachable!()
        }
    }
    pub fn binding_mut(&mut self) -> &mut TypeId {
        if let Type::Parameter {
            ref mut binding, ..
        } = self
        {
            binding
        } else {
            unreachable!()
        }
    }
    pub fn element_type(&self) -> TypeId {
        match self {
            Type::Array { typ, .. } => *typ,
            Type::Pointer { typ, .. } => *typ,
            Type::Type { ty, .. } => *ty,
            _ => unreachable!("can only get element types for pointers and arrays"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum T {
    None,
    Never,
    Void,
    Any,
    Boolean,

    I8,
    I16,
    I32,
    I64,
    CI8,
    CI16,
    CI32,
    CI64,

    U8,
    U16,
    U32,
    U64,
    CU8,
    CU16,
    CU32,
    CU64,

    F32,
    F64,

    IntegerLiteral,

    Type,
    Array,
    Pointer,

    String,

    Count,

    // Prelude types
    PointerU8,
    // String = T::Count as isize + 4,
}
