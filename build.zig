const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();

    const lib = b.addStaticLibrary("bolts", "src/bolts.zig");
    lib.setBuildMode(mode);
    lib.install();

    const dbg = b.addExecutable("dbg", "src/main.zig");
    dbg.addPackagePath("bolts", "src/bolts.zig");
    dbg.install();

    const run_cmd = dbg.run();
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const dbg_step = b.step("dbg", "Run debug application");
    dbg_step.dependOn(&run_cmd.step);

    const tests = b.addTest("src/bolts.zig");
    tests.setBuildMode(mode);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&tests.step);
}
