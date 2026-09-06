//! Env-gated statistics of the trace IR a native compile lowers.
//!
//! `OXIMOND_JIT_CENSUS=1` prints one `jit-census: {json}` line on stderr per
//! compile: per member the operations by variant, guards, snapshot entries,
//! values live at each guard and peak register pressure, and for the whole
//! compile the CLIF instructions each operation kind lowered to. Nothing here
//! changes generated code. With the variable unset, the only cost is the flag
//! load in [`enabled`].
//!
//! Positions follow `jit_ir::trace::opportunity`: preheader instructions
//! first, then the body, then the terminator at `end`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::jit::OptimizationTier;
use crate::jit_ir::trace::{
    RegisterEntry, TraceIr, TraceOp, TraceTerminator, TraceType, ValueDef, ValueId,
};

/// Whether the census is on.
pub(super) fn enabled() -> bool {
    crate::jit::debug_env::census()
}

/// The variant name of `op`, and for operations that carry an `op:` field
/// (arithmetic, numeric, native calls) `Variant.Sub`.
pub(super) fn op_name(op: &TraceOp) -> String {
    let text = format!("{op:?}");
    let end = text
        .find(|c: char| !c.is_ascii_alphanumeric())
        .unwrap_or(text.len());
    let mut name = text[..end].to_string();
    if let Some(at) = text.find(" op: ") {
        let rest = &text[at + 5..];
        let stop = rest
            .find(|c: char| !c.is_ascii_alphanumeric())
            .unwrap_or(rest.len());
        name.push('.');
        name.push_str(&rest[..stop]);
    }
    name
}

/// Statistics of one member (root or side) of a tree.
#[derive(Debug, Default, PartialEq)]
pub struct MemberCensus {
    pub preheader: usize,
    pub body: usize,
    pub values: usize,
    pub loop_params: usize,
    pub snapshots: usize,
    /// Peak count of live non-constant values, integer-class and float.
    pub peak_gpr: u32,
    pub peak_fpr: u32,
    pub ops: BTreeMap<String, u64>,
    /// Per guard: window entries, value, const, unchanged, root, virtual,
    /// slice entries, distinct non-constant values the snapshot names, values
    /// live across the guard that real uses keep, and snapshot values beyond
    /// those (weak uses).
    pub guards: Vec<[u32; 10]>,
}

/// Computes the census of one trace.
#[must_use]
pub fn member_census(ir: &TraceIr) -> MemberCensus {
    let pre = ir.preheader.len();
    let end = pre + ir.body.len();
    let n = ir.values.len();
    let mut census = MemberCensus {
        preheader: pre,
        body: ir.body.len(),
        values: n,
        loop_params: ir.loop_params.len(),
        snapshots: ir.snapshots.len(),
        ..MemberCensus::default()
    };
    let mut is_const = vec![false; n];
    let mut def: Vec<usize> = vec![0; n];
    for (position, instruction) in ir.preheader.iter().chain(&ir.body).enumerate() {
        *census.ops.entry(op_name(&instruction.op)).or_default() += 1;
        if let Some(result) = instruction.result
            && result.index() < n
        {
            def[result.index()] = position;
            is_const[result.index()] = matches!(instruction.op, TraceOp::Constant(_));
        }
    }
    // Last real use. An input or preheader value read in the body of a member
    // that loops back stays live through the back edge. A loop parameter is
    // redefined at the header each iteration, so a read ends its life unless
    // a `LoopBack` argument passes it on.
    let loops = matches!(ir.terminator, TraceTerminator::LoopBack { .. });
    let mut last: Vec<Option<usize>> = vec![None; n];
    let note = |value: ValueId, position: usize, last: &mut Vec<Option<usize>>| {
        if let Some(slot) = last.get_mut(value.index()) {
            *slot = Some(slot.map_or(position, |old| old.max(position)));
        }
    };
    for (position, instruction) in ir.preheader.iter().chain(&ir.body).enumerate() {
        instruction.op.operands(|operand| {
            let outside = loops
                && operand.index() < n
                && position >= pre
                && matches!(
                    ir.values[operand.index()].def,
                    ValueDef::Input(_) | ValueDef::FrameInput(_) | ValueDef::Preheader(_)
                );
            note(operand, if outside { end } else { position }, &mut last);
        });
    }
    let args: &[ValueId] = match &ir.terminator {
        TraceTerminator::LoopBack { args } | TraceTerminator::TailCall { args, .. } => args,
        _ => &[],
    };
    for arg in args {
        note(*arg, end, &mut last);
    }
    if let TraceTerminator::Return { value: Some(value) } = &ir.terminator {
        note(*value, end, &mut last);
    }
    // Difference arrays over positions 0..=end: pressure counts values live
    // at a position (defined at or before it, last used at or after it),
    // across counts those that survive it.
    let size = end + 2;
    let mut pressure = [vec![0i64; size], vec![0i64; size]];
    let mut across = vec![0i64; size];
    for v in 0..n {
        if is_const[v] {
            continue;
        }
        let Some(last_use) = last[v] else { continue };
        let class = usize::from(ir.values[v].ty == TraceType::Float);
        let start = match ir.values[v].def {
            ValueDef::Instruction(_) | ValueDef::Preheader(_) => def[v],
            ValueDef::LoopParam(_) => pre,
            ValueDef::Input(_) | ValueDef::FrameInput(_) => 0,
        };
        let last_use = last_use.max(start);
        pressure[class][start] += 1;
        pressure[class][last_use + 1] -= 1;
        if start < last_use {
            across[start + 1] += 1;
            across[last_use] -= 1;
        }
    }
    let peak = |diff: &[i64]| {
        let mut live = 0;
        let mut top = 0;
        for delta in diff {
            live += delta;
            top = top.max(live);
        }
        u32::try_from(top).unwrap_or(u32::MAX)
    };
    census.peak_gpr = peak(&pressure[0]);
    census.peak_fpr = peak(&pressure[1]);
    let mut live_across = vec![0i64; size];
    let mut running = 0;
    for (slot, delta) in live_across.iter_mut().zip(&across) {
        running += delta;
        *slot = running;
    }
    for (position, instruction) in ir.preheader.iter().chain(&ir.body).enumerate() {
        let Some(guard) = instruction.op.guard() else {
            continue;
        };
        let Some(info) = ir.guards.get(guard.index()) else {
            continue;
        };
        let Some(snapshot) = ir.snapshots.get(info.snapshot.index()) else {
            continue;
        };
        let mut counts = [0u32; 10];
        for frame in &snapshot.frames {
            for entry in &frame.registers {
                counts[0] += 1;
                let slot = match entry {
                    RegisterEntry::Value(_) => 1,
                    RegisterEntry::Const(_) => 2,
                    RegisterEntry::Unchanged => 3,
                    RegisterEntry::Root(_) => 4,
                    RegisterEntry::Virtual(_) => 5,
                    RegisterEntry::Slice(_) => 6,
                };
                counts[slot] += 1;
            }
        }
        let mut seen = vec![];
        snapshot.for_each_value(|value| {
            let v = value.index();
            if v < n && !is_const[v] && !seen.contains(&v) {
                seen.push(v);
            }
        });
        counts[7] = u32::try_from(seen.len()).unwrap_or(u32::MAX);
        let strong = |v: usize| {
            last[v].is_some_and(|l| l > position)
                && (def[v] < position
                    || !matches!(
                        ir.values[v].def,
                        ValueDef::Instruction(_) | ValueDef::Preheader(_)
                    ))
        };
        counts[9] = u32::try_from(seen.iter().filter(|v| !strong(**v)).count()).unwrap_or(u32::MAX);
        counts[8] = u32::try_from(live_across[position].max(0)).unwrap_or(u32::MAX);
        census.guards.push(counts);
    }
    census
}

#[derive(Default)]
struct Clif {
    /// Per operation kind: lowered instructions, CLIF instructions emitted
    /// by the lowering itself (children excluded).
    ops: BTreeMap<String, (u64, u64)>,
    /// CLIF emitted by lowerings that nest inside another (a side lowered
    /// inside its parent's instruction).
    stack: Vec<u64>,
}

thread_local! {
    static CLIF: RefCell<Clif> = RefCell::new(Clif::default());
}

/// Starts the CLIF accounting of one compile on this thread.
pub(super) fn begin() {
    CLIF.with(|c| *c.borrow_mut() = Clif::default());
}

/// Called before one operation lowers.
pub(super) fn enter() {
    CLIF.with(|c| c.borrow_mut().stack.push(0));
}

/// Called after it lowered `total` CLIF instructions, nested ones included.
pub(super) fn exit(op: &TraceOp, total: usize) {
    let name = op_name(op);
    CLIF.with(|c| {
        let mut c = c.borrow_mut();
        let children = c.stack.pop().unwrap_or(0);
        let total = total as u64;
        let own = total.saturating_sub(children);
        if let Some(parent) = c.stack.last_mut() {
            *parent += total;
        }
        let row = c.ops.entry(name).or_default();
        row.0 += 1;
        row.1 += own;
    });
}

/// The JSON line for one compile.
pub(super) fn report(
    members: &[&TraceIr],
    tier: OptimizationTier,
    clif_total: usize,
    detached: bool,
) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "{{\"root\":\"{:?}\",\"tier\":\"{tier:?}\",\"detached\":{detached},\"clif_total\":{clif_total},\"clif_ops\":{{",
        members.first().map(|ir| ir.id)
    );
    CLIF.with(|c| {
        for (i, (name, (count, clif))) in c.borrow().ops.iter().enumerate() {
            let _ = write!(
                out,
                "{}\"{name}\":[{count},{clif}]",
                if i > 0 { "," } else { "" }
            );
        }
    });
    out.push_str("},\"members\":[");
    for (i, ir) in members.iter().enumerate() {
        let m = member_census(ir);
        let _ = write!(
            out,
            "{}{{\"pre\":{},\"body\":{},\"values\":{},\"loop_params\":{},\"snapshots\":{},\"peak_gpr\":{},\"peak_fpr\":{},\"ops\":{{",
            if i > 0 { "," } else { "" },
            m.preheader,
            m.body,
            m.values,
            m.loop_params,
            m.snapshots,
            m.peak_gpr,
            m.peak_fpr
        );
        for (j, (name, count)) in m.ops.iter().enumerate() {
            let _ = write!(out, "{}\"{name}\":{count}", if j > 0 { "," } else { "" });
        }
        out.push_str("},\"guards\":[");
        for (j, g) in m.guards.iter().enumerate() {
            let _ = write!(out, "{}{g:?}", if j > 0 { "," } else { "" });
        }
        out.push_str("]}");
    }
    out.push_str("]}");
    out
}
