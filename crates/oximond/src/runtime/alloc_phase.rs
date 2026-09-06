//! Allocation phases for the `alloc-phases` diagnostic build.
//!
//! Each thread has a current [`Phase`]. The counting allocator of
//! `oximond-cli` reads it on every allocation and attributes the allocation
//! to it. Code marks a phase with [`enter`], which returns a guard that puts
//! the previous phase back when it drops: on return, on `?` and on unwind.
//! Phases nest, and the innermost one wins.
//!
//! A thread starts in [`Phase::Unclassified`]. The CLI moves its main thread
//! to [`Phase::Execution`] when guest code starts and to [`Phase::Report`]
//! when it writes reports. The VM and the JIT broker enter the recording,
//! compiler, installation and exit-recovery phases around that work, on
//! whichever thread does it.
//!
//! Without the `alloc-phases` feature [`enter`] returns a guard with no
//! state and no destructor, and [`current`] always says
//! [`Phase::Unclassified`]: the calls compile to nothing.

#[cfg(feature = "alloc-phases")]
use std::cell::Cell;

/// What a thread is doing when it allocates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Phase {
    /// Startup before guest code runs (state and library setup, loading the
    /// chunk), and any thread outside every other phase.
    Unclassified = 0,
    /// The trace recorder: starting, extending, finishing or aborting a
    /// recording, and submitting it to the broker.
    Recording = 1,
    /// Trace IR verification, tree planning and optimization.
    Optimization = 2,
    /// Exit planning and Cranelift code generation, up to the finished
    /// compiled trace.
    Codegen = 3,
    /// Installing a compiled trace into the VM, and dropping what it
    /// replaces.
    Installation = 4,
    /// Guest code: the interpreters, native traces and the runtime work they
    /// call, including garbage collection.
    Execution = 5,
    /// Rebuilding interpreter state after a native trace exits.
    ExitRecovery = 6,
    /// Writing the counter and statistics reports at exit.
    Report = 7,
}

impl Phase {
    /// Number of phases.
    pub const COUNT: usize = 8;

    /// Every phase, in index order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Unclassified,
        Self::Recording,
        Self::Optimization,
        Self::Codegen,
        Self::Installation,
        Self::Execution,
        Self::ExitRecovery,
        Self::Report,
    ];

    /// The phase's name in reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unclassified => "unclassified",
            Self::Recording => "recording",
            Self::Optimization => "optimization",
            Self::Codegen => "codegen",
            Self::Installation => "installation",
            Self::Execution => "execution",
            Self::ExitRecovery => "exit_recovery",
            Self::Report => "report",
        }
    }

    /// The phase with index `index`, or [`Phase::Unclassified`] when it is
    /// out of range.
    #[must_use]
    pub const fn from_index(index: u8) -> Self {
        if (index as usize) < Self::COUNT {
            Self::ALL[index as usize]
        } else {
            Self::Unclassified
        }
    }
}

#[cfg(feature = "alloc-phases")]
thread_local! {
    // A const-initialized `Copy` cell has no destructor and no lazy
    // registration, so the allocator can read it without allocating, even
    // while the thread exits.
    static CURRENT: Cell<u8> = const { Cell::new(Phase::Unclassified as u8) };
}

/// This thread's current phase.
#[inline]
#[must_use]
pub fn current() -> Phase {
    #[cfg(feature = "alloc-phases")]
    {
        CURRENT
            .try_with(Cell::get)
            .map_or(Phase::Unclassified, Phase::from_index)
    }
    #[cfg(not(feature = "alloc-phases"))]
    {
        Phase::Unclassified
    }
}

/// Makes `phase` this thread's current phase until the guard drops.
#[inline]
#[must_use = "the phase ends when the guard drops"]
pub fn enter(phase: Phase) -> PhaseGuard {
    #[cfg(feature = "alloc-phases")]
    {
        let previous = CURRENT.with(|current| current.replace(phase as u8));
        PhaseGuard { previous }
    }
    #[cfg(not(feature = "alloc-phases"))]
    {
        let _ = phase;
        PhaseGuard {}
    }
}

/// Restores the phase that was current before [`enter`].
pub struct PhaseGuard {
    #[cfg(feature = "alloc-phases")]
    previous: u8,
}

impl PhaseGuard {
    /// Ends the phase before the end of the scope.
    #[inline]
    pub fn end(self) {}
}

#[cfg(feature = "alloc-phases")]
impl Drop for PhaseGuard {
    #[inline]
    fn drop(&mut self) {
        CURRENT.with(|current| current.set(self.previous));
    }
}

#[cfg(all(test, feature = "alloc-phases"))]
mod tests {
    use super::{Phase, current, enter};

    #[test]
    fn guards_nest_and_restore_on_unwind() {
        assert_eq!(current(), Phase::Unclassified);
        {
            let _outer = enter(Phase::Execution);
            {
                let _inner = enter(Phase::Recording);
                assert_eq!(current(), Phase::Recording);
            }
            assert_eq!(current(), Phase::Execution);
            let caught = std::panic::catch_unwind(|| {
                let _codegen = enter(Phase::Codegen);
                panic!("unwind through the guard");
            });
            assert!(caught.is_err());
            assert_eq!(current(), Phase::Execution);
        }
        assert_eq!(current(), Phase::Unclassified);
    }

    #[test]
    fn every_thread_starts_unclassified() {
        let _here = enter(Phase::Execution);
        let there = std::thread::spawn(current).join().unwrap();
        assert_eq!(there, Phase::Unclassified);
    }
}
