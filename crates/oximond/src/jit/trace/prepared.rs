//! Objects a trace prepares once when it compiles, instead of on every call
//! of a helper: the validated pattern of a `string.find` or `string.match`
//! with a constant pattern, or the parsed format string of a `string.format`.
//!
//! A [`Prepared`] is built on the compile thread, so it is `Send + Sync` and
//! holds only owned data. The compiled code embeds the object's address as
//! an immediate argument of a specialized helper (see
//! [`super::helpers::prepared_helper_address`]), and the code's
//! [`super::emit::TraceCode`] owns the `Arc` for as long as that code can
//! run.

use crate::pattern::{GsubPlan, Pattern, RunShape, Template};
use crate::stdlib::PreparedFormat;
use std::sync::Arc;

/// A prepared object a compiled trace owns.
#[derive(Clone)]
pub enum Prepared {
    /// A validated pattern, anchors honored (see [`Pattern::new`]).
    Pattern(Arc<Pattern>),
    /// A validated `string.gmatch` pattern (see [`Pattern::new_for_gmatch`]).
    Gmatch(Arc<Pattern>),
    /// A `string.gsub` pattern and its replacement template.
    Gsub(Arc<GsubPlan>),
    /// A `string.format` format string, parsed.
    Format(Arc<PreparedFormat>),
    /// The class of a single-run pattern, which the trace scans itself.
    RunClass(Arc<RunClass>),
}

/// The class of a single-run pattern as a table by byte value (1 for a
/// member), which the emitted code reads, and the run's shape (see
/// [`Pattern::run_class`]).
pub struct RunClass {
    pub table: [u8; 256],
    pub shape: RunShape,
}

const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    send_sync::<Prepared>();
};

impl Prepared {
    /// The pattern of `bytes`, or `None` for one that fails to validate: the
    /// call then keeps the generic helper, which raises through the
    /// interpreter as before.
    #[must_use]
    pub fn pattern(bytes: &[u8]) -> Option<Self> {
        Pattern::new(bytes)
            .ok()
            .map(|pattern| Self::Pattern(Arc::new(pattern)))
    }

    /// The `string.gmatch` pattern of `bytes`, or `None` as for
    /// [`Self::pattern`].
    #[must_use]
    pub fn gmatch(bytes: &[u8]) -> Option<Self> {
        Pattern::new_for_gmatch(bytes)
            .ok()
            .map(|pattern| Self::Gmatch(Arc::new(pattern)))
    }

    /// The `string.gsub` plan for the pattern `bytes` and the replacement
    /// `template`, or `None` when the pattern fails to validate. A template
    /// that does not parse leaves the plan without one.
    #[must_use]
    pub fn gsub(bytes: &[u8], template: Option<&[u8]>) -> Option<Self> {
        let pattern = Pattern::new(bytes).ok()?;
        Some(Self::Gsub(Arc::new(GsubPlan {
            pattern,
            template: template.and_then(Template::parse),
        })))
    }

    /// The plan of the format string `bytes`, or `None` for one that does not
    /// parse.
    #[must_use]
    pub fn format(bytes: &[u8]) -> Option<Self> {
        PreparedFormat::new(bytes).map(|plan| Self::Format(Arc::new(plan)))
    }

    /// The class of the pattern `bytes`, or `None` for a pattern that is not
    /// a single run or fails to validate.
    #[must_use]
    pub fn run_class(bytes: &[u8]) -> Option<Self> {
        let (table, shape) = Pattern::new(bytes).ok()?.run_class()?;
        Some(Self::RunClass(Arc::new(RunClass {
            table: *table,
            shape,
        })))
    }

    /// The address a specialized helper receives. Valid while this object, or
    /// a clone of it, lives.
    #[must_use]
    pub fn address(&self) -> usize {
        match self {
            Self::Pattern(pattern) | Self::Gmatch(pattern) => Arc::as_ptr(pattern) as usize,
            Self::Gsub(plan) => Arc::as_ptr(plan) as usize,
            Self::Format(plan) => Arc::as_ptr(plan) as usize,
            Self::RunClass(class) => class.table.as_ptr() as usize,
        }
    }
}
