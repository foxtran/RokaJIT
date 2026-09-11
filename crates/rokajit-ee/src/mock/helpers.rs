use rokajit_ffi as ffi;

use super::MockEe;
use crate::ee_info::{HelperTarget, Helpers};
use crate::enums::CorInfoHelpFunc;
use crate::handles::{ClassHandle, MethodHandle};

impl Helpers for MockEe {
    fn get_helper_ftn(&self, _id: CorInfoHelpFunc) -> HelperTarget {
        HelperTarget {
            entrypoint: unsafe { std::mem::zeroed() },
            method: None,
        }
    }

    fn get_new_helper(
        &self,
        _token: &ffi::CORINFO_RESOLVED_TOKEN,
        _caller: MethodHandle,
    ) -> (CorInfoHelpFunc, Option<bool>) {
        (CorInfoHelpFunc::NEWFAST, Some(true))
    }

    fn get_casting_helper(
        &self,
        _token: &ffi::CORINFO_RESOLVED_TOKEN,
        throwing: bool,
    ) -> CorInfoHelpFunc {
        if throwing {
            CorInfoHelpFunc::CHKCASTANY
        } else {
            CorInfoHelpFunc::ISINSTANCEOFANY
        }
    }

    fn get_box_helper(&self, _cls: ClassHandle) -> CorInfoHelpFunc {
        CorInfoHelpFunc::BOX
    }

    fn get_function_entry_point(&self, _ftn: MethodHandle) -> ffi::CORINFO_CONST_LOOKUP {
        unsafe { std::mem::zeroed() }
    }
}
