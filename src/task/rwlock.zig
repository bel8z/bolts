const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

pub const RwLock = if (builtin.os.tag == .windows) Win32RwLock else std.Thread.RwLock;

const Win32RwLock = struct {
    const win32 = std.os.windows;
    const Self = @This();

    rwl: win32.SRWLOCK = win32.SRWLOCK_INIT,

    /// Attempts to obtain exclusive lock ownership.
    /// Returns `true` if the lock is obtained, `false` otherwise.
    pub fn tryLock(self: *Self) bool {
        return TryAcquireSRWLockExclusive(&self.rwl) != win32.FALSE;
    }

    /// Blocks until exclusive lock ownership is acquired.
    pub fn lock(self: *Self) void {
        AcquireSRWLockExclusive(&self.rwl);
    }

    /// Releases a held exclusive lock.
    /// Asserts the lock is held exclusively.
    pub fn unlock(self: *Self) void {
        ReleaseSRWLockExclusive(&self.rwl);
    }

    /// Attempts to obtain shared lock ownership.
    /// Returns `true` if the lock is obtained, `false` otherwise.
    pub fn tryLockShared(self: *Self) bool {
        return TryAcquireSRWLockShared(&self.rwl) != win32.FALSE;
    }

    /// Blocks until shared lock ownership is acquired.
    pub fn lockShared(self: *Self) void {
        AcquireSRWLockShared(&self.rwl);
    }

    /// Releases a held shared lock.
    pub fn unlockShared(self: *Self) void {
        ReleaseSRWLockShared(&self.rwl);
    }

    extern "kernel32" fn TryAcquireSRWLockExclusive(s: *win32.SRWLOCK) callconv(win32.WINAPI) win32.BOOLEAN;
    extern "kernel32" fn AcquireSRWLockExclusive(s: *win32.SRWLOCK) callconv(win32.WINAPI) void;
    extern "kernel32" fn ReleaseSRWLockExclusive(s: *win32.SRWLOCK) callconv(win32.WINAPI) void;
    extern "kernel32" fn TryAcquireSRWLockShared(s: *win32.SRWLOCK) callconv(win32.WINAPI) win32.BOOLEAN;
    extern "kernel32" fn AcquireSRWLockShared(s: *win32.SRWLOCK) callconv(win32.WINAPI) void;
    extern "kernel32" fn ReleaseSRWLockShared(s: *win32.SRWLOCK) callconv(win32.WINAPI) void;
};

fn testRwLock(comptime Rw: type) !void {
    const thread_count: usize = 4;
    const iter_count: usize = 1_000_000;
    const shared_length: usize = 8;

    const Ctx = struct {
        lock: Rw = .{},
        shared: [shared_length]i32 = [_]i32{0} ** shared_length,
        success: std.atomic.Atomic(bool),

        pub fn init() @This() {
            var ctx = @This(){ .success = std.atomic.Atomic(bool).init(true) };

            for (ctx.shared) |*x, i| {
                x.* = @intCast(i32, i);
            }

            return ctx;
        }

        pub fn threadFn(ctx: *@This(), id: usize) void {
            var prng = std.rand.DefaultPrng.init(0);
            const random = prng.random();

            _ = id;

            ctx.lock.lock();
            ctx.lock.unlock();

            var iter: usize = 0;
            while (iter < iter_count) : (iter += 1) {
                if (random.intRangeAtMost(u2, 0, 3) == 0) {
                    // Write an incrementing sequence of numbers (backwards).
                    var value = random.int(i32);

                    ctx.lock.lock();
                    defer ctx.lock.unlock();

                    for (ctx.shared) |_, i| {
                        const j = ctx.shared.len - i - 1;
                        ctx.shared[j] = value;
                        value -= 1;
                    }
                } else {
                    // Check that the sequence of numbers is incrementing.
                    var ok = true;
                    {
                        ctx.lock.lockShared();
                        defer ctx.lock.unlockShared();

                        var value = ctx.shared[0];
                        for (ctx.shared[1..]) |x| {
                            value += 1;
                            ok = ok and (value == x);
                        }
                    }

                    if (!ok) ctx.success.store(false, .Monotonic);
                }
            }
        }
    };

    var ctx = Ctx.init();
    var threads: [thread_count]std.Thread = undefined;

    ctx.lock.lock();
    for (threads) |*thread, id| {
        thread.* = try std.Thread.spawn(.{}, Ctx.threadFn, .{ &ctx, id });
    }

    var timer = try std.time.Timer.start();

    ctx.lock.unlock();
    for (threads) |thread| {
        thread.join();
    }

    std.debug.print("\nElapsed {} nanoseconds\n", .{timer.read()});

    try std.testing.expect(ctx.success.load(.Monotonic));
}

test "RwLock - Specialized" {
    try testRwLock(RwLock);
}

test "RwLock - Std" {
    try testRwLock(std.Thread.RwLock);
}
