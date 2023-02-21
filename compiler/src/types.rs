#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    None,
    Never,
    Void,
    Any,
    Boolean,
    // IntegerLiteral,
    // String,
    Type,

    Array {
        typ: TypeId,
        length: usize,
        is_generic: bool,
    },
    Function {
        parameters: Vec<TypeId>,
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
    Numeric {
        literal: bool,
        floating: bool,
        signed: bool,
        bytes: u8,
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

pub fn float_type(literal: bool, bytes: u8) -> Type {
    Type::Numeric {
        literal,
        floating: true,
        signed: true,
        bytes,
    }
}
impl Type {
    pub fn parameters(&self) -> &Vec<TypeId> {
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
    pub fn fields(&self) -> &Vec<TypeId> {
        if let Type::Tuple { fields } = self {
            fields
        } else {
            unreachable!()
        }
    }
    pub fn binding(&self) -> TypeId {
        if let Type::Parameter { binding, .. } = self {
            *binding
        } else {
            unreachable!()
        }
    }
    pub fn element_type(&self) -> TypeId {
        match self {
            Type::Array { typ, .. } => *typ,
            Type::Pointer { typ, .. } => *typ,
            _ => unreachable!("can only get element types for pointers and arrays"),
        }
    }
}

pub type TypeId = usize;
pub type TypeRef = Option<TypeId>;

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
