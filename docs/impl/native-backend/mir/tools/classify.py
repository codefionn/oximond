"""Shared phase classifier for perf samples and malloc stacks.

A stack is a list of demangled function names, leaf first. `phase_of` skips
allocator and libc leaf frames (so their cost goes to the caller's phase) and
returns the first frame that matches a rule. `is_alloc` says whether the leaf
side of the stack is allocator code.
"""
import re

ALLOC_LEAVES = re.compile(
    r"^(malloc|free|calloc|realloc|_int_malloc|_int_free|_int_realloc|malloc_consolidate|"
    r"cfree|__libc_malloc|__libc_free|__libc_realloc|__libc_calloc|__rust_alloc|__rust_dealloc|"
    r"__rust_realloc|__rust_alloc_zeroed|posix_memalign|aligned_alloc|_int_memalign|"
    r"unlink_chunk|tcache_.*|sysmalloc|alloc::alloc::.*|alloc::raw_vec::.*|"
    r"<alloc::alloc::Global.*|<std::alloc::System.*|std::sys::alloc::.*|"
    r"alloc::alloc::exchange_malloc|alloc::alloc::Global::.*|mallocprof.*|record|"
    r"__memcpy.*|__memmove.*|__memset.*|memcpy|memmove|memset|"
    r"__GI___.*|__GI_.*|_dl_.*|\[unknown\])$"
)

# (phase, regex over "function name  source path"). The first rule that
# matches a frame, scanning from the leaf, wins. A frame is the demangled
# function plus its file when known (addr2line), or the demangled path only
# (perf). Rules therefore name both crates ("regalloc2::") and paths
# ("regalloc2-0.15.2/").
RULES = [
    ("cl.regalloc2", r"regalloc2"),
    ("cl.emission", r"machinst[:/]+buffer|MachBuffer|isa[:/]+x64[:/]+inst[:/]+emit|cranelift-assembler|cranelift_assembler|x64[:/]+inst[:/]+emit"),
    ("cl.isle_lowering", r"constructor_|isle|isa[:/]+x64[:/]+lower|machinst[:/]+(lower|isle|abi|vcode|compile|blockorder|reg|inst_common)|isa[:/]+x64[:/]+(inst|abi|mod)|X64Backend|VCode|Lower<|isa[:/]+aarch64|isa[:/]+x64"),
    ("cl.module_setup", r"cranelift-jit|cranelift_jit|cranelift-module|cranelift_module|cranelift-native|cranelift_native|JITModule|JITBuilder|isa[:/]+(mod|lookup)\b|settings|Flags::new|OwnedTargetIsa|make_signature|declare_function|define_function|memmap|region::"),
    ("cl.clif_build_dfg", r"cranelift-frontend|cranelift_frontend|cranelift-codegen|cranelift_codegen|cranelift-entity|cranelift_entity|cranelift-control|cranelift-bforest|cranelift_bforest"),
    ("ox.clif_lowerer", r"jit[:/]+trace[:/]+emit|Lowerer|shared_stubs|SharedStubs"),
    ("ox.plan_exits", r"plan_exits|ExitPlan|exit_plan"),
    ("ox.verify", r"verify_trace|verify[:/]|TraceIr>::verify|verify_grown|::verify\b|fn verify|verify.rs"),
    ("ox.plan_tree", r"plan_tree|TreeMember|TreeWrites|carry_register|trace[:/]+tree"),
    ("ox.opt.finish_draft", r"Draft"),
    ("ox.opt.copies", r"eliminate_copies"),
    ("ox.opt.fold", r"fold_constants"),
    ("ox.opt.guards", r"eliminate_guards"),
    ("ox.opt.forward", r"forward_loads"),
    ("ox.opt.number", r"number_values"),
    ("ox.opt.concat", r"sink_concat_lengths"),
    ("ox.opt.slices", r"sink_slices"),
    ("ox.opt.dce", r"eliminate_dead_code"),
    ("ox.opt.hoist", r"hoist_invariants"),
    ("ox.opt.other", r"jit_ir[:/]+trace[:/]+(optimize|range|opportunity|helper|semantics)|optimize_tree_member|optimize_pipeline|optimize_with"),
    ("ox.compile_misc", r"jit[:/]+trace[:/]+compile|compile_with_plan|compile_side|compile_trace|CompiledTrace|jit[:/]+broker"),
    ("ox.install", r"install_compiled|install_tree|install_side|complete_compile|note_submitted|service_trace_compilations|publish"),
    ("ox.recording", r"recorder|Recorder|TracePolicy|record_|submit_queued|jit_ir[:/]+trace[:/]+ir|vm[:/]+jit[:/]+trace"),
    ("vm.native_trace", r"run_bound|run_checked|run_pinned|enter_lent|call_side|run_with_tables|run_sides"),
    ("vm.interpreter", r"oximond|vm[:/]|runtime[:/]|stdlib"),
]
COMPILED = [(name, re.compile(rx)) for name, rx in RULES]


def _strip_generics(name):
    """Drop generic arguments (`Vec<cranelift_codegen::X>`) but keep the impl
    type of a method (`<Lowerer>::foo`), so a Vec of a Cranelift type is not
    taken for Cranelift code."""
    def plain(text):
        out, depth = [], 0
        for ch in text:
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth = max(depth - 1, 0)
            elif depth == 0:
                out.append(ch)
        return "".join(out)

    if name.startswith("<"):
        d = 0
        for i, ch in enumerate(name):
            d += ch == "<"
            d -= ch == ">"
            if d == 0:
                return plain(name[1:i]) + plain(name[i + 1:])
    return plain(name)


def _split(frame):
    if isinstance(frame, tuple):
        return frame[0], _strip_generics(frame[0]) + " " + frame[1]
    return frame, _strip_generics(frame)


COMPILE_ANCHOR = re.compile(r"compile_with_plan|compile_side_with_plan|compile_trace_with_plan|CompilationJob")
INSTALL_ANCHOR = re.compile(r"complete_compile")
SUBMIT_ANCHOR = re.compile(r"service_trace_compilations")
RECORD_ANCHOR = re.compile(r"TracePolicy>::record\b|finish_closed|record_fast_hook|submit_queued|take_queued_tree|seed_plan")
PLAN_ANCHOR = re.compile(r"plan_tree|seed_plan|take_queued_tree")
IN_COMPILE_SKIP = {"ox.recording", "vm.native_trace", "vm.interpreter", "ox.install"}


def phase_of(frames):
    """frames: leaf-first list of function names or (function, file) pairs.

    Frames are first reduced to a region by looking for an anchor anywhere in
    the stack: a trace compile, an install, or a submission. Inside a compile
    the leaf-first rule scan skips the recorder and interpreter rules, so a
    `TraceIr::clone` is the compile's, not the recorder's.
    """
    funcs = [_split(f) for f in frames]
    region = None
    for func, full in funcs:
        if COMPILE_ANCHOR.search(func):
            region = "compile"
            break
        if INSTALL_ANCHOR.search(func) and region is None:
            region = "install"
        if SUBMIT_ANCHOR.search(func) and region is None:
            region = "submit"
        if RECORD_ANCHOR.search(func) and region is None:
            region = "record"
    if region == "record" and any(PLAN_ANCHOR.search(func) for func, _ in funcs):
        return "ox.plan_tree"
    for func, full in funcs:
        if ALLOC_LEAVES.match(func):
            continue
        for phase, rx in COMPILED:
            if region == "compile" and phase in IN_COMPILE_SKIP:
                continue
            if region == "install" and not phase.startswith(("cl.", "ox.install")):
                continue
            if region in ("submit", "record") and phase.startswith(("vm.", "cl.")):
                continue
            if rx.search(full):
                return phase
    return {"compile": "ox.compile_glue", "install": "ox.install", "submit": "ox.submit", "record": "ox.recording"}.get(region, "other")


def is_alloc(frames):
    return bool(frames) and bool(ALLOC_LEAVES.match(_split(frames[0])[0]))
