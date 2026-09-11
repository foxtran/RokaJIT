use super::MockEe;
use crate::ee_info::ClassQueries;
use crate::enums::{ClassAttribs, CorInfoType};
use crate::handles::ClassHandle;

impl ClassQueries for MockEe {
    fn as_cor_info_type(&self, _cls: ClassHandle) -> CorInfoType {
        CorInfoType::Class
    }

    fn is_value_class(&self, _cls: ClassHandle) -> bool {
        false
    }

    fn get_class_attribs(&self, _cls: ClassHandle) -> ClassAttribs {
        self.class_attribs
    }

    fn get_class_size(&self, _cls: ClassHandle) -> u32 {
        8
    }

    fn get_type_for_primitive_numeric_class(&self, _cls: ClassHandle) -> Option<CorInfoType> {
        None
    }
}
