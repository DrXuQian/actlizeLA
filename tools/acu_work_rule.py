"""ACU --import rule: every captured action and native PC, without filtering.

Only imports supplied by old ACU's embedded Python (no external acu_report,
torch or json/gzip dependency). The parent checks bindings and completion.
"""
import csv
import os

import HgRules

FIELDS = (
    ("Address", "address"), ("Function Id", "function_id"),
    ("Source", "sass_code"), ("Instructions Executed", "warp_executed_count"),
    ("Thread Instructions Executed", "thread_executed_count"),
)
OPTIONAL = (
    "vector_live_registers", "scalar_live_registers", "achieved_access_cachelines",
    "theoretical_access_cachelines", "achieved_access_sectors", "achieved_access_data_bytes",
    "source_file_name", "source_file_line_number",
)
STALLS = (
    "instruction_fetch", "compute_dependency", "memory_dependency", "memory_throttle",
    "not_selected", "pipe_busy", "sleeping", "sync", "salu_control", "simt_control",
    "sreg_read", "fence_sys", "amc", "others", "wait_pc", "total",
)
BINDINGS = (
    "Range Id", "Function Name", "Kernel Name", "Kernel Mangled Name", "Grid Size", "Block Size",
    "Duration", "Duration Unit", "Instruction Metric", "Aggregate Instructions", "Instruction Unit",
)


def get_identifier():
    return "ActlizeGdnWork"


def get_name():
    return "Actlize GDN complete native work"


def get_description():
    return "Offline report actions and native PC counts; no kernel execution"


def apply(handle):
    path = HgRules.get_output_file_path(handle)
    if not path or path == "-":
        raise ValueError("explicit --stats-rule-output file required")
    metric_id = os.environ["ACTLIZE_ACU_INSTRUCTION_METRIC"]
    context = HgRules.get_context(handle)
    names = getattr(HgRules.IAction, "NameBase", HgRules.IAction)
    header = list(BINDINGS) + [label for label, _ in FIELDS] + list(OPTIONAL)
    header += [group + "_" + reason for group in ("stall", "not_issued_stall") for reason in STALLS]
    actions, pcs = 0, 0
    with open(path, "x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, header, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for i in range(context.num_ranges()):
            report_range = context.range_by_idx(i)
            for j in range(report_range.num_actions()):
                action = report_range.action_by_idx(j)
                native, functions = action.sass(), action.functions()
                if not native:
                    raise ValueError("missing native instructions: " + action.name())
                duration = action.metric_by_name("ppu__time_duration.sum")
                instructions = action.metric_by_name(metric_id)
                if not duration.has_value() or not instructions.has_value():
                    raise ValueError("missing recorded duration or instruction counter")
                binding = dict(zip(BINDINGS, (
                    str(action.capture_sequence_id()), "", action.name(names.NameBase_DEMANGLED),
                    action.name(names.NameBase_MANGLED), action.grid_size(), action.block_size(),
                    duration.value(), duration.unit(), metric_id, instructions.value(), instructions.unit(),
                )))
                for pc, insn in sorted(native.items()):
                    if insn.address != pc:
                        raise ValueError("native PC address mismatch")
                    row = dict(binding, **{label: getattr(insn, field) for label, field in FIELDS})
                    row["Address"] = hex(pc)
                    row["Function Name"] = functions[insn.function_id]
                    # Missing optional diagnostics stay blank, never synthetic 0.
                    row.update({field: getattr(insn, field, None) for field in OPTIONAL})
                    for group in ("stall", "not_issued_stall"):
                        reason = getattr(insn, group, None)
                        row.update({group + "_" + field: getattr(reason, field, None) for field in STALLS})
                    writer.writerow(row)
                    pcs += 1
                actions += 1
    if not actions or not pcs:
        raise ValueError("empty native report")
    # Old ACU can return rc0 after a Python failure. Partial CSV is not success.
    with open(path + ".complete", "x", encoding="utf-8") as complete:
        complete.write("%d,%d\n" % (actions, pcs))
